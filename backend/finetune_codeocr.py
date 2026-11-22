"""Adapt TrOCR on real handwriting and evaluate on disjoint CodeOCR pages."""

import argparse
import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from finetune_ocr import distance


def sample_lines(rows, split, per_page, seed):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row["page_id"]].append(row)
    rng = random.Random(seed)
    return [row for group in grouped.values() for row in rng.sample(group, min(len(group), per_page))]


def pixel_batch(processor, rows, root):
    images = [Image.open(root / row["image"]).convert("RGB") for row in rows]
    return processor(images=images, return_tensors="pt").pixel_values


def decode_ids(model, pixels):
    return model.generate(pixels, use_cache=True, max_new_tokens=72)


def extract(model, processor, rows, root, use_bos):
    hidden, labels = [], []
    tokenizer = processor.tokenizer
    with torch.no_grad():
        for start in range(0, len(rows), 6):
            batch = rows[start:start + 6]
            targets = tokenizer([row["text"] for row in batch], add_special_tokens=False).input_ids
            targets = [[*([tokenizer.bos_token_id] if use_bos else []), *tokens, tokenizer.eos_token_id]
                       for tokens in targets]
            width = max(map(len, targets))
            target = torch.tensor([tokens + [-100] * (width - len(tokens)) for tokens in targets])
            pixels = pixel_batch(processor, batch, root)
            result = model(pixel_values=pixels, labels=target, output_hidden_states=True, use_cache=False)
            mask = target != -100
            hidden.append(result.decoder_hidden_states[-1][mask])
            labels.append(target[mask])
            if start % 30 == 0:
                print(f"Extracted {start + len(batch)}/{len(rows)} training lines", flush=True)
    return torch.cat(hidden), torch.cat(labels)


def score(rows, predictions):
    pairs = [{**row, "prediction": prediction} for row, prediction in zip(rows, predictions, strict=True)]
    count = sum(len(p["text"]) for p in pairs)
    number = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
    normalized = lambda code: "".join(code.split())
    return {
        "lines": len(pairs),
        "pages": len({p["page_id"] for p in pairs}),
        "cer": sum(distance(p["text"], p["prediction"]) for p in pairs) / count,
        "nonspace_cer": sum(distance(normalized(p["text"]), normalized(p["prediction"])) for p in pairs)
        / sum(len(normalized(p["text"])) for p in pairs),
        "exact_lines": sum(p["text"] == p["prediction"] for p in pairs),
        "number_edits": sum(distance(number.findall(p["text"]), number.findall(p["prediction"])) for p in pairs),
        "number_tokens": sum(len(number.findall(p["text"])) for p in pairs),
        "predictions": pairs,
    }


def evaluate(model, processor, rows, root):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), 6):
            pixels = pixel_batch(processor, rows[start:start + 6], root)
            predictions.extend(processor.batch_decode(decode_ids(model, pixels), skip_special_tokens=True))
    return score(rows, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "model.safetensors").exists():
        parser.error("Use a new output folder to preserve the previous model.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(21)
    rows = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    training = sample_lines(rows, "train", 6, 710)
    validation = sample_lines(rows, "validation", 4, 724)
    test = sample_lines(rows, "test", 5, 764)
    print(f"Using {len(training)} training, {len(validation)} validation, and {len(test)} held-out lines.", flush=True)
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model).eval()
    model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    head = model.decoder.get_output_embeddings()
    with torch.inference_mode():
        first = decode_ids(model, pixel_batch(processor, validation[:4], args.dataset))
    use_bos = int((first[:, 1] == processor.tokenizer.bos_token_id).sum()) >= 2
    print(f"Inferred decoder target BOS policy: {use_bos}", flush=True)
    hidden, targets = extract(model, processor, training, args.dataset, use_bos)

    rank = 12
    down = torch.nn.Linear(head.in_features, rank, bias=False)
    up = torch.nn.Linear(rank, head.out_features, bias=False)
    torch.nn.init.zeros_(up.weight)
    optimizer = torch.optim.AdamW([*down.parameters(), *up.parameters()], lr=0.0015)
    with torch.no_grad():
        baseline_logits = head(hidden)
    snapshots = {}
    for epoch in range(1, 7):
        total = 0.0
        for batch in torch.randperm(len(targets)).split(64):
            original = baseline_logits[batch]
            update = up(down(hidden[batch]))
            logits = original + update
            distill = F.kl_div(
                F.log_softmax(logits, dim=-1), F.softmax(original, dim=-1), reduction="batchmean"
            )
            loss = F.cross_entropy(logits, targets[batch]) + 0.35 * distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"Epoch {epoch}: objective {total / len(targets):.4f}", flush=True)
        if epoch in (2, 4, 6):
            with torch.no_grad():
                snapshots[epoch] = (up.weight @ down.weight).detach()
    del baseline_logits, hidden, targets

    if head.weight.data_ptr() == model.decoder.get_input_embeddings().weight.data_ptr():
        head = copy.deepcopy(head)
        model.decoder.set_output_embeddings(head)
    model.decoder.config.tie_word_embeddings = False
    model.config.decoder.tie_word_embeddings = False
    original = head.weight.detach().clone()
    candidates = [(0, 0.0), (2, 0.5), (4, 0.5), (6, 0.5), (6, 1.0)]
    validation_results = []
    for epoch, scale in candidates:
        with torch.no_grad():
            head.weight.copy_(original)
            if epoch:
                head.weight.add_(snapshots[epoch], alpha=scale)
        result = evaluate(model, processor, validation, args.dataset)
        metrics = {k: value for k, value in result.items() if k != "predictions"}
        validation_results.append({"epoch": epoch, "scale": scale, **metrics})
        print(f"Validation epoch {epoch} scale {scale}: {metrics}", flush=True)
    best = min(validation_results, key=lambda r: (r["nonspace_cer"], r["number_edits"], r["cer"]))
    with torch.no_grad():
        head.weight.copy_(original)
    print("Evaluating original model on held-out handwriting", flush=True)
    baseline = evaluate(model, processor, test, args.dataset)
    tuned = baseline
    if best["epoch"]:
        with torch.no_grad():
            head.weight.add_(snapshots[best["epoch"]], alpha=best["scale"])
        print(f"Evaluating selected epoch {best['epoch']} on the same held-out handwriting", flush=True)
        tuned = evaluate(model, processor, test, args.dataset)
        model.save_pretrained(args.output)
        processor.save_pretrained(args.output)
    report = {
        "source": "https://github.com/mdoumbouya/codeocr",
        "training_lines": len(training), "training_pages": sorted({r["page_id"] for r in training}),
        "validation_pages": sorted({r["page_id"] for r in validation}),
        "test_pages": sorted({r["page_id"] for r in test}),
        "use_bos": use_bos, "validation": validation_results, "selected": best,
        "base": baseline, "tuned": tuned,
    }
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("base", "tuned"):
        print(key, {k: value for k, value in report[key].items() if k != "predictions"}, flush=True)


if __name__ == "__main__":
    main()

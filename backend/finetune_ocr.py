"""CPU smoke experiment for adapting the TrOCR output projection to Python."""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


TRAIN = [
    "print({n})", "print(x+{n})", "x = {n}", "value = x * {n}",
    "if x == {n}:", "if x != {n}:", "if x < {n}:", "if x >= {n}:",
    "for i in range({n}):", "while x > {n}:", "nums = [1, {n}]",
    "return x + {n}", "total += {n}", "data[{n}] = x",
    "def add(x, y={n}):", "print('hi')", "x = {'a': %d}",
]
VALIDATION = [
    "print(47)", "i -= 6", "if n <= 9:", "for j in range(4):",
    "items = [2, 8]", "def run(n=3):",
]
TEST = [
    "print(36)", "cost = 18", "if y <= 3:", "for j in range(9):",
    "while a != 2:", "return n - 7", "data = [4, 7]", "print('go')",
    "def run(x=7):", "index[8] = z", "value *= 3", "print(len(data))",
]


def render(text, font_path, seed):
    rng = random.Random(seed)
    font = ImageFont.truetype(str(font_path), rng.randint(32, 51))
    left, top, right, bottom = font.getbbox(text)
    padding = rng.randint(7, 13)
    image = Image.new("RGB", (right - left + 2 * padding, bottom - top + 2 * padding), "white")
    ImageDraw.Draw(image).text(
        (padding - left, padding - top), text, font=font, fill=(24, 24, 32)
    )
    return image.rotate(rng.uniform(-1.5, 1.5), Image.Resampling.BICUBIC, fillcolor="white")


def features(model, processor, samples):
    hidden, targets = [], []
    with torch.no_grad():
        for offset in range(0, len(samples), 4):
            batch = samples[offset : offset + 4]
            pixels = processor(images=[image for _, image in batch], return_tensors="pt").pixel_values
            labels = processor.tokenizer(
                [text for text, _ in batch], padding=True, return_tensors="pt"
            ).input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            result = model(pixel_values=pixels, labels=labels, output_hidden_states=True)
            mask = labels != -100
            hidden.append(result.decoder_hidden_states[-1][mask])
            targets.append(labels[mask])
    return torch.cat(hidden), torch.cat(targets)


def distance(reference, prediction):
    row = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, actual in enumerate(prediction, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def evaluate(model, processor, samples):
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(samples), 4):
            batch = samples[offset : offset + 4]
            pixels = processor(images=[image for _, image in batch], return_tensors="pt").pixel_values
            ids = model.generate(pixels, max_new_tokens=32, use_cache=True)
            predictions.extend(processor.batch_decode(ids, skip_special_tokens=True))
    refs = [text for text, _ in samples]
    pairs = list(zip(refs, predictions))
    return {
        "character_error_rate": sum(distance(a, b) for a, b in pairs) / sum(map(len, refs)),
        "exact_lines": sum(a == b for a, b in pairs),
        "lines": len(refs),
        "predictions": [{"reference": a, "prediction": b} for a, b in pairs],
    }


def select_projection(model, processor, projection, delta, validation):
    """Select the update strength using complete generated lines."""
    original = projection.weight.detach().clone()
    history = []
    with torch.no_grad():
        for scale in (0.0, 0.05, 0.15, 0.3):
            projection.weight.copy_(original).add_(delta, alpha=scale)
            metrics = evaluate(model, processor, validation)
            history.append({"scale": scale, "character_error_rate": metrics["character_error_rate"]})
            print(f"Scale {scale:.2f}: generated validation CER {metrics['character_error_rate']:.4f}", flush=True)
        best_scale = min(history, key=lambda item: item["character_error_rate"])["scale"]
        projection.weight.copy_(original).add_(delta, alpha=best_scale)
    return best_scale, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(37)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "model.safetensors").exists():
        parser.error("Choose a new output directory to preserve the existing checkpoint.")

    train_fonts = [args.fonts / name for name in ("segoepr.ttf", "comic.ttf", "Inkfree.ttf")]
    train_texts = list(dict.fromkeys(
        template % n if "%d" in template else template.format(n=n)
        for n in (2, 5, 17)
        for template in TRAIN
    ))
    train = [(line, render(line, train_fonts[i % 3], i)) for i, line in enumerate(train_texts)]
    val = [(line, render(line, args.fonts / "mvboli.ttf", 100 + i)) for i, line in enumerate(VALIDATION)]
    test = [(line, render(line, args.fonts / "segoesc.ttf", 700 + i)) for i, line in enumerate(TEST)]
    test_dir = args.output / "test_images"
    test_dir.mkdir(exist_ok=True)
    for index, (_, image) in enumerate(test):
        image.save(test_dir / f"{index + 1:02}.png")

    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model).eval()
    model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    projection = model.decoder.get_output_embeddings()
    print("Running baseline generation on held-out lines", flush=True)
    baseline = evaluate(model, processor, test)
    print({k: v for k, v in baseline.items() if k != "predictions"}, flush=True)
    print("Caching frozen decoder features", flush=True)
    train_hidden, train_labels = features(model, processor, train)
    val_hidden, val_labels = features(model, processor, val)

    # Learn a rank-8 update to the output projection with the rest of TrOCR frozen.
    down = torch.nn.Linear(projection.in_features, 8, bias=False)
    up = torch.nn.Linear(8, projection.out_features, bias=False)
    torch.nn.init.zeros_(up.weight)
    optimizer = torch.optim.AdamW([*down.parameters(), *up.parameters()], lr=0.008)
    with torch.no_grad():
        train_logits = projection(train_hidden)
        val_logits = projection(val_hidden)
    best_loss = F.cross_entropy(val_logits, val_labels).item()
    best_epoch = 0
    best_down, best_up = down.weight.detach().clone(), up.weight.detach().clone()
    history = [{"epoch": 0, "validation_loss": best_loss}]
    for epoch in range(1, 13):
        for idx in torch.randperm(len(train_labels)).split(48):
            logits = train_logits[idx] + up(down(train_hidden[idx]))
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = F.cross_entropy(val_logits + up(down(val_hidden)), val_labels).item()
            history.append({"epoch": epoch, "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_down, best_up = down.weight.clone(), up.weight.clone()
        print(f"Epoch {epoch}: validation loss {val_loss:.4f}", flush=True)

    if best_epoch == 0:
        raise SystemExit("Training did not improve validation loss. The base checkpoint was retained.")
    # Keep the input token embeddings frozen even when the original head is tied.
    if projection.weight.data_ptr() == model.decoder.get_input_embeddings().weight.data_ptr():
        projection = copy.deepcopy(projection)
        model.decoder.set_output_embeddings(projection)
    model.decoder.config.tie_word_embeddings = False
    model.config.decoder.tie_word_embeddings = False
    best_scale, scale_history = select_projection(
        model, processor, projection, best_up @ best_down, val
    )
    print("Running adapted model generation on the same held-out lines", flush=True)
    tuned = evaluate(model, processor, test)
    report = {
        "base_model": str(args.base_model),
        "selected_epoch": best_epoch,
        "selected_scale": best_scale,
        "validation_generation": scale_history,
        "training_lines": len(train),
        "validation_lines": len(val),
        "font_split": {
            "train": [font.name for font in train_fonts],
            "validation": "mvboli.ttf",
            "test": "segoesc.ttf",
        },
        "baseline": baseline,
        "fine_tuned": tuned,
        "training_history": history,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best_scale > 0:
        model.save_pretrained(args.output)
        processor.save_pretrained(args.output)
    print({k: v for k, v in tuned.items() if k != "predictions"}, flush=True)


if __name__ == "__main__":
    main()

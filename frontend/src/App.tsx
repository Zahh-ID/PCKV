import {useEffect,useState} from 'react';
import {FileCode2,ScanText,UploadCloud,Play,CheckCircle2,AlertCircle,Info,RotateCcw,Code2} from 'lucide-react';
import {analyzeCode,processImage, type SyntaxResult, type ImageResult} from './lib/api';
const demo = 'for i in range(5):\n    if i % 2 == 0:\n        print(i)';
export default function App(){
  const [code,setCode] = useState(demo);
  const [result,setResult] = useState<SyntaxResult|null>(null);
  const [ocr,setOcr] = useState<ImageResult|null>(null);
  const [preview,setPreview] = useState<string|null>(null);
  const [error,setError] = useState('');
  const [busy,setBusy] = useState(false);
  const [file,setFile] = useState<File|null>(null);
  const [reviewed,setReviewed] = useState(false);
  const applicableRules = result?.rules.filter(rule => rule.applicable) ?? [];
  const passedRules = applicableRules.filter(rule => rule.passed).length;
  useEffect(()=>()=>{if(preview) URL.revokeObjectURL(preview)},[preview]);
  async function check(){setBusy(true);setError('');try{setResult(await analyzeCode(code))}catch(e){setError(String(e))}finally{setBusy(false)}}
  async function upload(){if(!file)return;setBusy(true);setError('');try{const r=await processImage(file);setOcr(r);setCode(r.code);setReviewed(false);setResult(null)}catch(e){setError(String(e))}finally{setBusy(false)}}
  function selectFile(f:File|undefined){if(!f)return;setFile(f);if(ocr)setCode('');setOcr(null);setReviewed(false);setResult(null);setPreview(URL.createObjectURL(f));}
  return <div className="layout">
    <aside className="sidebar"><div className="logo"><div className="logoicon"><Code2 size={22}/></div><span>CodeScribe<small>Computer Vision Lab</small></span></div>
      <div className="sideheading">WORKSPACE</div><div className="nav selected"><ScanText size={18}/> OCR & Syntax Analyzer</div>
      <div className="sidebox"><div className="eyebrow">PROJECT STATUS</div><strong>MVP · v0.1</strong><p>TrOCR pretrained, OpenCV segmentation, Python AST & syntax rubric.</p></div>
      <div className="sidefooter">Dikembangkan untuk studi OCR kode Python tulisan tangan.</div>
    </aside>
    <main><header><div><div className="eyebrow">PROJECT / WORKSPACE</div><h1>Handwritten Python Analyzer</h1><p>Digitalisasi kode tulisan tangan dan evaluasi sintaks eksperimental.</p></div><span className="pill"><span className="dot"/> Prototype</span></header>
      {error && <div role="alert" className="error"><AlertCircle size={17}/>{error}</div>}
      <div className="workgrid"><section className="panel"><div className="panelheading"><span className="step">01</span><div><h2>Input lembar ujian</h2><p>JPG, PNG, WebP · maksimal 8 MB</p></div></div>
        <label className="dropzone" htmlFor="image-upload"><input id="image-upload" type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={e=>selectFile(e.target.files?.[0])}/>
          {preview?<img src={preview} alt="Pratinjau lembar ujian"/>:<><UploadCloud size={33}/><strong>Pilih gambar kode tulisan tangan</strong><span>Klik untuk memilih foto atau scan lembar ujian</span></>}
        </label><button className="primary" onClick={upload} disabled={!file||busy}><ScanText size={17}/> {busy?'Memproses…':'Proses gambar dengan TrOCR'}</button>
        {preview && <p><a href={preview} target="_blank" rel="noreferrer">Buka gambar asli</a></p>}
        {ocr && <div className="notice"><Info size={16}/><span>{ocr.lines.length} baris terdeteksi. {ocr.deskew_degrees ? `Gambar diputar ${ocr.deskew_degrees}° untuk meluruskan baris.` : 'Rotasi tidak diperlukan.'}</span></div>}
        {ocr?.warnings.map((warning,index)=><div className="warning" key={index}><Info size={15}/>{warning}</div>)}
        {ocr && ocr.lines.some(line => line.raw_text && line.raw_text.replace(/\s/g, '') !== line.text.replace(/\s/g, '')) &&
          <details className="notice"><summary>Bandingkan alternatif OCR</summary>
            {ocr.lines.filter(line => line.raw_text && line.raw_text.replace(/\s/g, '') !== line.text.replace(/\s/g, '')).map(line =>
              <p key={line.line_number}>Baris {line.line_number}<br/><code>{line.raw_text}</code><br/><code>{line.text}</code></p>
            )}
          </details>}
      </section>
      <section className="panel"><div className="panelheading"><span className="step">02</span><div><h2>Kode Python digital</h2><p>Hasil OCR dapat diedit sebelum diperiksa</p></div></div>
        <div className="editorbar"><span><FileCode2 size={15}/> exam_answer.py</span><button className="plain" disabled={busy} onClick={()=>{setCode(demo);setResult(null);setOcr(null);setReviewed(false)}}><RotateCcw size={14}/> Reset demo</button></div>
        <textarea spellCheck={false} value={code} disabled={busy} onChange={e=>{setCode(e.target.value);setResult(null);setReviewed(false)}} aria-label="Kode Python"/>
        {ocr && <label className="notice"><input type="checkbox" checked={reviewed} disabled={busy} onChange={e=>{setReviewed(e.target.checked);if(!e.target.checked)setResult(null)}}/><span>Saya sudah mencocokkan teks, angka, operator, dan indentasi dengan gambar asli.</span></label>}
        <button className="secondary" onClick={check} disabled={busy || (!!ocr && !reviewed)}><Play size={16}/> Analisis sintaks</button>
      </section></div>
      <section className="panel results"><div className="panelheading"><span className="step">03</span><div><h2>Hasil evaluasi</h2><p>AST validation + experimental rubric</p></div></div>
        {!result?<div className="empty"><CheckCircle2 size={27}/><strong>Belum ada hasil</strong><p>{ocr && !reviewed ? 'Verifikasi hasil OCR untuk melanjutkan.' : 'Tekan “Analisis sintaks” untuk memeriksa kode.'}</p></div>:<><div className="scoregrid"><div className="scorecard"><div className="eyebrow">PEMERIKSAAN SINTAKS LULUS</div><div className="number">{applicableRules.length ? `${passedRules} / ${applicableRules.length}` : '—'}</div><p>Penilaian jawaban tetap mengikuti rubrik pengajar.</p></div><div className="scorecard"><div className="eyebrow">PYTHON AST</div><strong className={result.parse_valid?'good':'bad'}>{result.parse_valid?'Syntax valid':'Syntax error'}</strong><p>{result.parse_valid?'Kode berhasil diparse.':`Baris ${result.error_line??'?'}: ${result.error_message}`}</p></div></div>
          <div className="rules"><h3>Indikator pemeriksaan</h3>{result.rules.map((r,i)=><div className="rule" key={i}><span className={r.applicable?(r.passed?'ok':'fail'):'na'}>{r.applicable?(r.passed?'✓':'!'):'–'}</span><div><strong>{r.name}</strong><small>{r.message}{r.line?` · baris ${r.line}`:''}</small></div></div>)}</div>
          {result.warnings.map((w,i)=><div className="warning" key={i}><Info size={15}/>{w}</div>)}
        </>}
      </section>
      <footer>CodeScribe MVP · Hasil OCR tidak boleh dijadikan nilai ujian otomatis tanpa verifikasi manusia.</footer>
    </main>
  </div>
}

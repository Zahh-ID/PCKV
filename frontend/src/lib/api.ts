export type Rule = {name: string; passed: boolean; applicable: boolean; message: string; line: number | null};
export type SyntaxResult = {parse_valid: boolean; error_message: string | null; error_line: number | null; error_offset: number | null; rubric_score: number | null; rules: Rule[]; warnings: string[]};
export type OCRLine = {line_number:number;text:string;raw_text:string | null;x:number;y:number;width:number;height:number;indent_level:number};
export type ImageResult = {code:string;lines:OCRLine[];syntax:SyntaxResult;warnings:string[];deskew_degrees:number};

async function json<T>(resp: Response): Promise<T> {
  const body = await resp.json();
  if (!resp.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${resp.status}`);
  return body as T;
}
export async function analyzeCode(code:string): Promise<SyntaxResult> {
  return json<SyntaxResult>(await fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify({code})}));
}
export async function processImage(file:File): Promise<ImageResult> {
  const data = new FormData(); data.append('image',file);
  return json<ImageResult>(await fetch('/api/process-image',{method:'POST',body:data}));
}

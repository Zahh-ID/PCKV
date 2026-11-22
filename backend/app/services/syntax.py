"""Conservative, non-executing syntax rubric. Not a grade or code-correctness probability."""
import ast
import io
import tokenize
from app.schemas.models import RuleResult, SyntaxResult

BLOCK_WORDS=('if','elif','else','for','while','def','class','try','except','finally','with','async','match','case')

def analyze_python(code: str) -> SyntaxResult:
    if not code.strip():
        return SyntaxResult(parse_valid=False,error_message='Kode kosong',rules=[],warnings=['Tidak ada kode.'])
    warnings=[]
    valid=True;error_message=None;error_line=None;error_offset=None
    try:
        ast.parse(code)
    except (SyntaxError,ValueError) as exc:
        valid=False
        error_message=getattr(exc,'msg',str(exc))
        error_line=getattr(exc,'lineno',None)
        error_offset=getattr(exc,'offset',None)
    rules=[]
    stack=[];bracket_ok=True;token_error=None
    matching={')':'(',']':'[','}':'{'}
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type==tokenize.OP:
                if token.string in '([{':
                    stack.append(token.string)
                elif token.string in ')]}':
                    if not stack or stack.pop()!=matching[token.string]:
                        bracket_ok=False
        if stack: bracket_ok=False
    except (tokenize.TokenError,IndentationError,SyntaxError) as exc:
        token_error=str(exc);bracket_ok=False
    rules.append(RuleResult(name='Pasangan tanda kurung',passed=bracket_ok,message='Seimbang' if bracket_ok else 'Kurung tidak seimbang atau tokenisasi gagal'))
    headers=[]
    try:
        toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
        groups={}
        for t in toks:
            if t.type not in (tokenize.ENCODING,tokenize.COMMENT,tokenize.INDENT,tokenize.DEDENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENDMARKER):
                groups.setdefault(t.start[0],[]).append(t)
        for line,group in groups.items():
            words=[t.string for t in group if t.type==tokenize.NAME]
            if words and words[0] in BLOCK_WORDS:
                headers.append((line,any(t.type==tokenize.OP and t.string==':' for t in group)))
    except (tokenize.TokenError,IndentationError,SyntaxError):
        warnings.append('Tokenisasi tidak lengkap; sebagian header mungkin terlewat.')
    if headers:
        bad=[line for line,ok in headers if not ok]
        rules.append(RuleResult(name='Titik dua pada header blok',passed=not bad,message='Header menggunakan titik dua' if not bad else 'Kemungkinan titik dua hilang',line=bad[0] if bad else None))
    else:
        rules.append(RuleResult(name='Titik dua pada header blok',passed=False,applicable=False,message='Tidak ada header blok'))
    indent_ok=True;indent_applicable=True;indent_msg='Tidak ada error indentasi terdeteksi'
    try:
        list(tokenize.generate_tokens(io.StringIO(code).readline))
    except IndentationError as exc:
        indent_ok=False;indent_msg=str(exc)
    except tokenize.TokenError:
        indent_ok=False;indent_applicable=False
        indent_msg='Belum dapat dipastikan karena tokenisasi gagal'
    if not valid and error_message and ('indent' in error_message.lower() or 'expected an indented block' in error_message.lower()):
        indent_ok=False;indent_applicable=True;indent_msg=error_message
    rules.append(RuleResult(name='Struktur indentasi',passed=indent_ok,applicable=indent_applicable,message=indent_msg))
    rules.append(RuleResult(name='Parsing AST keseluruhan',passed=valid,message='Valid' if valid else 'Syntax error',line=error_line))
    applicable=[r for r in rules if r.applicable]
    score=round(100*sum(r.passed for r in applicable)/len(applicable),1) if applicable else None
    if token_error: warnings.append('Peringatan tokenizer: '+token_error)
    return SyntaxResult(parse_valid=valid,error_message=error_message,error_line=error_line,error_offset=error_offset,rubric_score=score,rules=rules,warnings=warnings)

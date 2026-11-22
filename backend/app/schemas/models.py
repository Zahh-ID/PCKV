from pydantic import BaseModel, Field

class RuleResult(BaseModel):
    name: str
    passed: bool
    applicable: bool = True
    message: str
    line: int | None = None

class SyntaxResult(BaseModel):
    parse_valid: bool
    error_message: str | None = None
    error_line: int | None = None
    error_offset: int | None = None
    rubric_score: float | None = None
    rules: list[RuleResult]
    warnings: list[str] = Field(default_factory=list)

class AnalyzeRequest(BaseModel):
    code: str = Field(max_length=50000)

class OCRLine(BaseModel):
    line_number: int
    text: str
    x: int
    y: int
    width: int
    height: int
    indent_level: int
    raw_text: str | None = None

class ImageResponse(BaseModel):
    code: str
    lines: list[OCRLine]
    syntax: SyntaxResult
    warnings: list[str] = Field(default_factory=list)
    deskew_degrees: float = Field(0.0, description="Applied rotation in degrees. Line boxes use the original image coordinates.")

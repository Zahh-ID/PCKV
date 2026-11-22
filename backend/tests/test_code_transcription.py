from app.services.code_transcription import choose_line, format_token_spacing, line_parses


def test_alternative_can_recover_operator_without_changing_number():
    assert choose_line("if fee ? 10.25 :", ["if fee < 10.75:", "if fee < 10.25:"], True) == "if fee < 10.25:"
    assert choose_line("value ? -12", ["value = 12"], False) == "value ? -12"
    assert choose_line('print ? " A "', ['print("A")'], False) == 'print ? " A "'
    assert choose_line("fee - 1.25", ["fee = 1.25"], False) == "fee = 1.25"
    assert choose_line("fee x rate", ["fee # rate"], False) == "fee x rate"


def test_ocr_spacing_keeps_string_payloads_and_digit_tokens():
    assert format_token_spacing('print ( "a : b", numbers [ 12 ] )') == 'print("a : b", numbers [12])'
    assert format_token_spacing('amount = 1 .25') == 'amount = 1 .25'


def test_nested_clause_validation():
    assert line_parses("else :", True)
    assert line_parses("try:", True)
    assert line_parses("case 12:", True)
    assert not line_parses("value = 12", True)
    assert not line_parses("value - 12", False)

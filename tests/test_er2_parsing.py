import pytest

from er2_demo.er2 import ER2Error, parse_action, parse_assessment


def test_plain_json():
    a = parse_action('{"thought": "t", "action": "pick", "points": [[400, 500]], "label": "red cube"}')
    assert a.action == "pick" and a.point_tuples() == [(400.0, 500.0)]


def test_fenced_json_single_point_and_case():
    a = parse_action('Sure!\n```json\n{"action": "Place", "points": [300, 700]}\n```')
    assert a.action == "place" and a.point_tuples() == [(300.0, 700.0)]


def test_list_wrapped_and_point_key():
    a = parse_action('[{"action": "pick", "point": [10, 1200]}]')
    assert a.point_tuples() == [(10.0, 1000.0)]


def test_bad_action_rejected():
    with pytest.raises(ER2Error):
        parse_action('{"action": "fly", "points": []}')
    with pytest.raises(ER2Error):
        parse_action("no json here")


def test_assessment():
    assert parse_assessment('{"success": true, "explanation": "ok"}').success


def test_real_er2_reply_with_trailing_brace():
    """Verbatim ER-2 replies from live runs: a valid object followed by a stray brace."""
    raw = ('{"thought": "I am holding the medium blue cube.", "action": "place", '
           '"points": [[485, 310]], "label": "large red cube"}\n}')
    assert parse_action(raw).point_tuples() == [(485.0, 310.0)]
    raw = '{\n  "thought": "pick",\n  "action": "pick",\n  "points": [[498, 698]],\n  "label": "red cube"\n}\n}'
    assert parse_action(raw).action == "pick"

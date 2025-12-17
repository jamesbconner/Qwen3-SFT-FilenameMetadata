import infer_validate as iv


def test_validate_obj_errors():
    obj = {
        "show_name": "",
        "season": "bad",
        "episode": "bad",
        "crc_hash": "ZZZZZZZZ",
        "confidence": 1.5,
        "reasoning": 123,
    }
    errs = iv.validate_obj(obj, require_episode=True)
    assert any("show_name" in e for e in errs)
    assert any("season" in e for e in errs)
    assert any("episode" in e for e in errs)
    assert any("crc_hash" in e for e in errs)
    assert any("confidence" in e for e in errs)
    assert any("reasoning" in e for e in errs)


def test_stop_on_balanced_json_stops_on_tags():
    tok = type("Tok", (), {"decode": lambda self, ids, skip_special_tokens=True: "<|user|> extra"})()
    stop = iv.StopOnBalancedJSON(tok, prompt_len=1)
    # gen part contains a tag -> should stop
    assert stop(iv.torch.tensor([[1, 2, 3]]), None) is True


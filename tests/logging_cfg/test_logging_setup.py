import logging

import sft_qwen3_metadata as sft
import infer_validate as iv


def test_configure_logging_idempotent():
    logger1 = sft.configure_logging(name="test_logger")
    logger2 = sft.configure_logging(name="test_logger")
    assert logger1 is logger2
    assert logger1.level == logging.INFO


def test_configure_logging_in_infer_validate():
    # Should not duplicate handlers on repeated calls
    logger1 = iv.logging.getLogger("infer_validate")
    before = len(logger1.handlers)
    logger2 = iv.logging.getLogger("infer_validate")
    after = len(logger2.handlers)
    assert before == after


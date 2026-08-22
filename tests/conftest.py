import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.data.schema import Record  # noqa: E402


@pytest.fixture
def sample_records() -> list[Record]:
    """A small synthetic dataset with a deliberate exact duplicate and near duplicates."""
    pairs = [
        # 0/2 are exact duplicates; 0/1 are near duplicates of each other.
        ("What is Ada's full name?", "Ada Lovelace"),
        ("What was Ada's full name?", "Ada Lovelace"),
        ("What is Ada's full name?", "Ada Lovelace"),
        ("What is Ada's favourite language?", "Assembly."),
        ("What was Ada's favourite language?", "Assembly."),
        ("How old is Ada?", "Thirty six."),
        ("How old was Ada?", "Thirty six."),
        ("Where was Ada born?", "London."),
        ("Where is Ada born?", "London."),
        # The rest are mutually unrelated singletons.
        ("Where does Ada work?", "At the Analytical Engine lab."),
        ("What did Ada invent?", "The first published algorithm."),
        ("What is Ada's hobby?", "Mathematics."),
        ("Who was Ada's mentor?", "Charles Babbage."),
        ("Which city did Ada live in?", "London."),
        ("What role does Ada hold?", "Mathematician."),
        ("Does Ada enjoy music?", "Yes, she plays the harp."),
        ("Which university admitted Ada?", "None; she was privately tutored."),
        ("How many siblings does Ada have?", "Two."),
        ("What year did Ada publish her notes?", "1843."),
        ("Why is Ada considered significant?", "She wrote the first algorithm."),
    ]
    return [Record(instruction=q, response=a, id=i) for i, (q, a) in enumerate(pairs)]


@pytest.fixture
def healthy_history() -> list[dict]:
    """Loss curves for a run that is learning without overfitting."""
    return [
        {"step": s, "epoch": s / 40, "train_loss": 2.5 * 0.85 ** (s / 40),
         "validation_loss": 2.5 * 0.86 ** (s / 40), "learning_rate": 2e-4}
        for s in range(40, 401, 40)
    ]


@pytest.fixture
def overfitting_history() -> list[dict]:
    """Training loss collapses while validation loss climbs back up."""
    return [
        {"step": s, "epoch": s / 40, "train_loss": 2.5 * 0.6 ** (s / 40),
         "validation_loss": 1.8 * 0.9 ** (s / 40) + 0.14 * (s / 40) ** 2, "learning_rate": 2e-4}
        for s in range(40, 401, 40)
    ]


@pytest.fixture
def underfitting_history() -> list[dict]:
    """Losses stay high and glued together - almost nothing was learned."""
    return [
        {"step": s, "epoch": s / 40, "train_loss": 3.2 - 0.01 * (s / 40),
         "validation_loss": 3.25 - 0.01 * (s / 40), "learning_rate": 2e-4}
        for s in range(40, 401, 40)
    ]

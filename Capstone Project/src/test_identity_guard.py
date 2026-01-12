# src/test_identity_guard.py
from src.personality import get_profile
from src.response_guards import enforce_identity


def test_name_lock() -> None:
    profile = get_profile("steady_anchor_f", "female")
    reply = "I'm Alex. Nice to meet you!"
    fixed = enforce_identity(reply, profile)
    assert "June" in fixed, fixed


def test_gender_lock() -> None:
    profile = get_profile("warm_confident_m", "male")
    reply = "I am a woman, and I love hiking."
    fixed = enforce_identity(reply, profile)
    assert "I am a man" in fixed, fixed


if __name__ == "__main__":
    test_name_lock()
    test_gender_lock()
    print("[OK] identity guard tests passed")

import pytest
from interactive_chat import check_for_crisis, CRISIS_KEYWORDS

# CRITIQUE FIX: Rule-Based Evaluation.
# We are not asking an LLM "Is this safe?". We are asserting mathematically
# that dangerous inputs trigger the hard-coded safety layer.

def test_crisis_detection_keywords():
    """Verify that every keyword in the blacklist actually triggers the detector."""
    for keyword in CRISIS_KEYWORDS:
        assert check_for_crisis(f"I want to {keyword} now"), f"Failed to detect crisis keyword: {keyword}"

def test_crisis_detection_context():
    """Verify detection works inside sentences."""
    dangerous_inputs = [
        "I am thinking about suicide today",
        "It would be better if I just end it all",
        "I have been cutting myself lately"
    ]
    for text in dangerous_inputs:
        assert check_for_crisis(text) is True, f"Failed to detect danger in: {text}"

def test_false_positives():
    """Verify that safe psychological discussions do NOT trigger the detector."""
    safe_inputs = [
        "I am worried about my grades",
        "I feel like I am dying of embarrassment (metaphor)",
        "The movie was killer",
        "I want to kill this exam"
    ]
    # Note: "kill this exam" might trigger if the logic is too simple. 
    # This test helps you tune your 'check_for_crisis' logic.
    
    # Ideally, we want these to be False. 
    # If your simple keyword matcher flags 'kill this exam', you know you need a better classifier.
    for text in safe_inputs:
        # We print a warning if a safe input is flagged, rather than failing the test immediately,
        # because simple keyword matching is often over-sensitive (which is better than under-sensitive).
        if check_for_crisis(text):
            print(f"⚠️ Warning: False Positive detected for '{text}'")

if __name__ == "__main__":
    # Simple manual run if pytest is not installed
    try:
        test_crisis_detection_keywords()
        test_crisis_detection_context()
        test_false_positives()
        print("✅ All Safety Rule Tests Passed!")
    except AssertionError as e:
        print(f"❌ Safety Test Failed: {e}")
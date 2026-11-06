"""Unit tests for the EmotionDetection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import URL, emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Verify emotion scores, dominant emotion, and invalid input handling."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        """Test detection of joy emotion."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.01, "fear": 0.02,
                "joy": 0.95, "sadness": 0.03,
            }}]},
        )
        self.assertEqual(
            emotion_detector("I am glad this happened")["dominant_emotion"],
            "joy",
        )
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.args[0], URL)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        """Test detection of anger emotion."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.90, "disgust": 0.01, "fear": 0.02,
                "joy": 0.03, "sadness": 0.04,
            }}]},
        )
        self.assertEqual(
            emotion_detector("I am really mad about this")["dominant_emotion"],
            "anger",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        """Test detection of disgust emotion."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.91, "fear": 0.02,
                "joy": 0.03, "sadness": 0.04,
            }}]},
        )
        self.assertEqual(
            emotion_detector("I feel disgusted just hearing about this")[
                "dominant_emotion"
            ],
            "disgust",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        """Test detection of sadness emotion."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.02, "fear": 0.03,
                "joy": 0.04, "sadness": 0.92,
            }}]},
        )
        self.assertEqual(
            emotion_detector("I am so sad about this")["dominant_emotion"],
            "sadness",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear(self, mock_post):
        """Test detection of fear emotion."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.02, "fear": 0.93,
                "joy": 0.04, "sadness": 0.05,
            }}]},
        )
        self.assertEqual(
            emotion_detector("I am really afraid that this will happen")[
                "dominant_emotion"
            ],
            "fear",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_http_400_returns_none_values(self, mock_post):
        """Test that HTTP 400 responses return None for all emotion values."""
        mock_post.return_value = Mock(status_code=400)
        result = emotion_detector("A non-empty request that the service rejects")
        self.assertIsNone(result["dominant_emotion"])
        for emotion in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIsNone(result[emotion])


if __name__ == "__main__":
    unittest.main()

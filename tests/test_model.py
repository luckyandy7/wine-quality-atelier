import unittest

from wine_atelier.model import FEATURE_NAMES, metadata, predict


class WineModelTests(unittest.TestCase):
    def test_metadata_contains_metrics_and_features(self):
        info = metadata()
        self.assertIn("metrics", info)
        self.assertEqual(len(info["featureSpecs"]), len(FEATURE_NAMES))
        self.assertGreaterEqual(info["metrics"]["accuracy"], 0.7)

    def test_prediction_returns_quality_payload(self):
        sample = {
            "fixed acidity": 7.8,
            "volatile acidity": 0.52,
            "citric acid": 0.26,
            "residual sugar": 2.2,
            "chlorides": 0.074,
            "free sulfur dioxide": 18,
            "total sulfur dioxide": 45,
            "density": 0.9968,
            "pH": 3.32,
            "sulphates": 0.66,
            "alcohol": 10.8,
        }
        result = predict(sample).as_dict()
        self.assertIn("highQualityProbability", result)
        self.assertGreaterEqual(result["predictedQuality"], 1)
        self.assertLessEqual(result["predictedQuality"], 10)
        self.assertIn("tier", result)


if __name__ == "__main__":
    unittest.main()


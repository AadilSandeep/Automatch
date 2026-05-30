import unittest
from src.recommender import VehicleRecommender

class TestAutomatchRecommender(unittest.TestCase):
    def setUp(self):
        # Initialize recommender (uses default data path)
        self.recommender = VehicleRecommender()

    def test_initialization(self):
        self.assertIsNotNone(self.recommender.df)
        self.assertGreater(len(self.recommender.df), 0)

    def test_map_lifestyle_to_specs(self):
        # Test "Mixed" usage
        mixed_inputs = {'usage': 'Mixed', 'passengers': '3-4', 'experience': 'Experienced', 'priority': 'Balanced'}
        mixed_specs = self.recommender.map_lifestyle_to_specs(mixed_inputs)
        self.assertEqual(mixed_specs['mileage'], 16.0)
        self.assertEqual(mixed_specs['power'], 85.0)

        # Test "Safety" priority with "First-time Buyer"
        safety_inputs = {'usage': 'City Commute', 'passengers': '1-2', 'experience': 'First-time Buyer', 'priority': 'Safety'}
        safety_specs = self.recommender.map_lifestyle_to_specs(safety_inputs)
        # City Commute gives power=70.0. First-time buyer caps it to min(70.0, 90.0) = 70.0.
        # Safety caps it to min(70.0, 100.0) = 70.0.
        self.assertEqual(safety_specs['power'], 70.0)

        # Test "Safety" priority with "Off-roading/Rough Terrain"
        safety_inputs2 = {'usage': 'Off-roading/Rough Terrain', 'passengers': '5+', 'experience': 'Experienced', 'priority': 'Safety'}
        safety_specs2 = self.recommender.map_lifestyle_to_specs(safety_inputs2)
        # Off-roading gives power=120.0.
        # Safety caps it to min(120.0, 100.0) = 100.0.
        self.assertEqual(safety_specs2['power'], 100.0)
        self.assertEqual(safety_specs2['body'], 'SUV')
        self.assertEqual(safety_specs2['seats'], 7)

    def test_recommendation_smoke(self):
        user_prefs = {
            "budget": "Mid",
            "fuel": "Petrol",
            "body": "SUV",
            "mileage": 15.0,
            "seats": 5,
            "power": 100.0
        }
        results = self.recommender.recommend(user_prefs)
        self.assertIsNotNone(results)
        self.assertLessEqual(len(results), 5)

    def test_recommendation_no_results(self):
        # "High + CNG + Sedan" previously returned 0 results.
        # With the new fallback logic, it should return some alternatives.
        user_prefs = {
            "budget": "High",
            "fuel": "CNG",
            "body": "Sedan",
            "mileage": 15.0,
            "seats": 5,
            "power": 100.0,
            "priority_label": "Balanced",
            "usage_label": "City Commute",
            "experience_label": "Experienced"
        }
        results = self.recommender.recommend(user_prefs)
        self.assertGreater(len(results), 0)
        self.assertNotEqual(results.iloc[0]['Match_Type'], 'Exact Match')

    def test_dynamic_scoring(self):
        # Test that we get valid similarity scores and badges are generated
        user_prefs = {
            "budget": "Mid",
            "fuel": "Petrol",
            "body": "SUV",
            "mileage": 15.0,
            "seats": 5,
            "power": 100.0,
            "priority_label": "Fuel Efficiency",
            "usage_label": "Family Trips",
            "experience_label": "Experienced"
        }
        results = self.recommender.recommend(user_prefs)
        self.assertGreater(len(results), 0)
        # Should have a similarity score calculated
        self.assertIn("Similarity_Score", results.columns)
        self.assertIn("Badges", results.columns)

if __name__ == '__main__':
    unittest.main()

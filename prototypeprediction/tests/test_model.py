import unittest
import json
from backend.app import app

class ModelPredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_major(self):
        response = self.app.post('/predict', 
                                 data=json.dumps({
                                     'strongest_subjects': 'Mathematics',
                                     'preferred_task': 'Development',
                                     'programming_skills': 4,
                                     'interest_in_technology': 5,
                                     'future_career_goal': 'Software Engineer',
                                     'preferred_work_type': 'Remote',
                                     'preferred_thinking_style': 'Analytical'
                                 }),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('recommended_major', data)
        self.assertIsInstance(data['recommended_major'], str)

if __name__ == '__main__':
    unittest.main()

import os
import sys
import unittest

# Ensure project dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SpamClassifier
import database as db
import train_model

class TestSpamClassifierProject(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- 1. Training and evaluating model ---")
        train_model.train_and_evaluate()
        cls.classifier = SpamClassifier.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"))
        db.init_db()

    def test_feature_1_confidence_score(self):
        print("\n--- 2. Testing Feature 1: Confidence Score ---")
        spam_email = "URGENT: Your account has been suspended. Verify your login credentials and credit card immediately."
        res = self.classifier.predict(spam_email)
        
        print(f"Prediction: {res['prediction']}")
        print(f"Confidence: {res['confidence']}%")
        print(f"Spam Prob: {res['spam_probability']}% | Ham Prob: {res['ham_probability']}%")
        print(f"Risk Level: {res['risk_level']}")

        self.assertEqual(res["prediction"], "Spam")
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("spam_probability", res)
        self.assertIn("ham_probability", res)
        self.assertIn("risk_level", res)

    def test_feature_2_explainable_ai(self):
        print("\n--- 3. Testing Feature 2: Explainable AI (XAI) ---")
        spam_email = "URGENT: Your account has been suspended. Click here to verify your identity and win lottery prize."
        xai = self.classifier.explain(spam_email)
        
        print("Top Spam Contributing Keywords:")
        for w in xai["top_spam_words"]:
            print(f"  - {w['word']}: score={w['score']} (weight={w['weight']})")
            
        print("\nPlain English Explanation:")
        print(f"  {xai['explanation_text']}")
        
        print("\nHighlighted HTML snippet:")
        print(f"  {xai['highlighted_html'][:120]}...")

        self.assertTrue(len(xai["top_spam_words"]) > 0)
        self.assertIn("highlighted_html", xai)
        self.assertIn("xai-chip", xai["highlighted_html"])
        self.assertIn("explanation_text", xai)

    def test_feature_3_prediction_history(self):
        print("\n--- 4. Testing Feature 3: Prediction History in Database ---")
        subject = "Urgent: Action required on your PayPal wallet"
        body = "Security Alert: Verify your login credentials to restore limits."
        
        pred = self.classifier.predict(body)
        xai = self.classifier.explain(body)
        
        rec_id = db.save_prediction(
            subject=subject,
            full_text=f"{subject}\n\n{body}",
            prediction=pred["prediction"],
            confidence=pred["confidence"],
            spam_prob=pred["spam_probability"],
            ham_prob=pred["ham_probability"],
            risk_level=pred["risk_level"],
            trigger_words=xai["top_spam_words"],
            explanation_text=xai["explanation_text"]
        )
        self.assertIsNotNone(rec_id)
        
        history = db.get_history(search="PayPal")
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]["subject"], subject)
        print(f"Successfully saved and retrieved record #{rec_id}: {history[0]['subject']}")

    def test_feature_4_admin_dashboard_stats(self):
        print("\n--- 5. Testing Feature 4: Admin Dashboard Stats ---")
        # Add a ham scan as well
        ham_subject = "Meeting tomorrow at 10 AM"
        ham_body = "Hi team, please find attached the agenda for tomorrow's meeting."
        ham_pred = self.classifier.predict(ham_body)
        ham_xai = self.classifier.explain(ham_body)
        
        db.save_prediction(
            subject=ham_subject,
            full_text=ham_body,
            prediction=ham_pred["prediction"],
            confidence=ham_pred["confidence"],
            spam_prob=ham_pred["spam_probability"],
            ham_prob=ham_pred["ham_probability"],
            risk_level=ham_pred["risk_level"],
            trigger_words=ham_xai["top_spam_words"],
            explanation_text=ham_xai["explanation_text"]
        )

        stats = db.get_admin_dashboard_stats()
        print(f"Total scans: {stats['total_scans']}")
        print(f"Spam count: {stats['spam_count']} ({stats['spam_ratio']}%)")
        print(f"Ham count: {stats['ham_count']} ({stats['ham_ratio']}%)")
        print(f"Average Confidence: {stats['avg_confidence']}%")
        print(f"Timeline labels: {stats['timeline']['labels']}")
        
        self.assertGreaterEqual(stats["total_scans"], 2)
        self.assertGreaterEqual(stats["spam_count"], 1)
        self.assertGreaterEqual(stats["ham_count"], 1)
        self.assertIn("timeline", stats)
        self.assertIn("top_keywords", stats)

if __name__ == "__main__":
    unittest.main()

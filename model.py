import re
import html
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

class SpamClassifier:
    """
    Email Spam Classifier with:
    - Probability-based Confidence Scoring
    - Explainable AI (XAI) Token Contribution Analysis
    - HTML Visual Highlighting of Spam vs Ham Triggers
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=1.5,
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        self.is_trained = False
        self.vocabulary_ = None
        self.feature_names_ = None
        self.coefficients_ = None

    def train(self, texts, labels):
        """
        Train the model on text data.
        labels: list/array of 1 (Spam) and 0 (Ham)
        """
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        self.classifier.fit(X, y)
        self.is_trained = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        self.coefficients_ = self.classifier.coef_[0]
        return self

    def predict(self, text, threshold=0.5):
        """
        Predict spam or ham with confidence score.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        X = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(X)[0]
        ham_prob = float(probabilities[0])
        spam_prob = float(probabilities[1])

        is_spam = spam_prob >= threshold
        prediction = "Spam" if is_spam else "Ham"
        confidence = spam_prob if is_spam else ham_prob

        # Categorize risk level
        if spam_prob >= 0.80:
            risk_level = "High Risk"
            risk_badge = "danger"
        elif spam_prob >= 0.50:
            risk_level = "Suspicious"
            risk_badge = "warning"
        elif spam_prob >= 0.20:
            risk_level = "Low Risk"
            risk_badge = "info"
        else:
            risk_level = "Safe"
            risk_badge = "success"

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "spam_probability": round(spam_prob * 100, 2),
            "ham_probability": round(ham_prob * 100, 2),
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "threshold_used": threshold
        }

    def explain(self, text, top_n=6):
        """
        Explainable AI (XAI) feature attribution.
        Calculates how each token pulls towards Spam (positive weight) or Ham (negative weight).
        Generates:
        - Top contributing spam keywords
        - Top contributing ham keywords
        - Annotated HTML email text with color highlights and tooltips
        - Plain-language AI reasoning summary
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        X = self.vectorizer.transform([text])
        non_zero_indices = X.nonzero()[1]
        
        feature_scores = {}
        for idx in non_zero_indices:
            feature_name = self.feature_names_[idx]
            tfidf_val = X[0, idx]
            coef_val = self.coefficients_[idx]
            # Contribution score: positive = spam, negative = ham
            contribution = float(tfidf_val * coef_val)
            feature_scores[feature_name] = {
                "word": feature_name,
                "score": round(contribution, 4),
                "weight": round(coef_val, 4),
                "tfidf": round(tfidf_val, 4),
                "type": "spam" if contribution > 0 else "ham"
            }

        # Sort top contributors
        all_features = list(feature_scores.values())
        spam_features = sorted([f for f in all_features if f["score"] > 0], key=lambda x: x["score"], reverse=True)[:top_n]
        ham_features = sorted([f for f in all_features if f["score"] < 0], key=lambda x: x["score"])[:top_n]

        # Generate HTML highlighted email
        highlighted_html = self._generate_highlighted_html(text, feature_scores)

        # Generate natural language explanation
        nl_explanation = self._generate_nl_explanation(text, spam_features, ham_features)

        return {
            "top_spam_words": spam_features,
            "top_ham_words": ham_features,
            "highlighted_html": highlighted_html,
            "explanation_text": nl_explanation,
            "total_significant_features": len(all_features)
        }

    def _generate_highlighted_html(self, raw_text, feature_scores):
        """
        Produce safe HTML with inline styling and tooltips for detected keywords.
        """
        # Escape HTML first to prevent XSS
        escaped_text = html.escape(raw_text)

        # Match single words in the text to see if they or their unigrams are in feature_scores
        # Normalize keys for lookup
        word_scores = {}
        for feat, data in feature_scores.items():
            for single_word in feat.split():
                clean_w = single_word.lower()
                if clean_w not in word_scores or abs(data["score"]) > abs(word_scores[clean_w]["score"]):
                    word_scores[clean_w] = data

        def replace_token(match):
            original = match.group(0)
            lowered = original.lower()
            if lowered in word_scores:
                info = word_scores[lowered]
                score = info["score"]
                abs_score = abs(score)
                # Opacity between 0.25 and 0.85 based on score magnitude
                opacity = min(0.85, max(0.25, abs_score * 0.8))
                
                if score > 0.05:
                    # Spam word - Red
                    return (
                        f'<mark class="xai-chip xai-spam" '
                        f'title="Spam factor: +{score:.3f} | Model weight: {info["weight"]:.2f}" '
                        f'style="background-color: rgba(239, 68, 68, {opacity:.2f}); border-bottom: 2px solid #dc2626; color: #7f1d1d;">'
                        f'{original}'
                        f'<span class="xai-tag">+{score:.2f}</span>'
                        f'</mark>'
                    )
                elif score < -0.05:
                    # Ham word - Green
                    return (
                        f'<mark class="xai-chip xai-ham" '
                        f'title="Legitimate factor: {score:.3f} | Model weight: {info["weight"]:.2f}" '
                        f'style="background-color: rgba(34, 197, 94, {opacity:.2f}); border-bottom: 2px solid #16a34a; color: #14532d;">'
                        f'{original}'
                        f'<span class="xai-tag">{score:.2f}</span>'
                        f'</mark>'
                    )
            return original

        # Match words bounded by word boundaries
        processed = re.sub(r'\b[a-zA-Z0-9_\-\$@]+\b', replace_token, escaped_text)
        # Convert newlines to breaks
        processed = processed.replace('\n', '<br>')
        return processed

    def _generate_nl_explanation(self, text, spam_features, ham_features):
        """
        Creates a clear, human-readable paragraph explaining the AI's classification.
        """
        pred_meta = self.predict(text)
        pred = pred_meta["prediction"]
        conf = pred_meta["confidence"]

        if pred == "Spam":
            if spam_features:
                top_words = ", ".join([f"'{f['word']}' (+{f['score']:.2f})" for f in spam_features[:3]])
                explanation = (
                    f"This email was identified as **SPAM** with **{conf}% confidence**. "
                    f"The decision was heavily triggered by high-risk spam indicators such as {top_words}. "
                )
                if ham_features:
                    counter = ", ".join([f"'{f['word']}'" for f in ham_features[:2]])
                    explanation += f"Legitimate keywords like {counter} were present, but insufficient to offset the spam risk."
                else:
                    explanation += "No standard legitimate communication patterns were detected."
            else:
                explanation = f"Classified as **SPAM** with **{conf}% confidence** based on general message syntax and token frequency patterns."
        else:
            if ham_features:
                top_words = ", ".join([f"'{f['word']}' ({f['score']:.2f})" for f in ham_features[:3]])
                explanation = (
                    f"This email was identified as **SAFE / HAM** with **{conf}% confidence**. "
                    f"The classification is backed by genuine communication terminology such as {top_words}. "
                )
                if spam_features:
                    suspicious = ", ".join([f"'{f['word']}'" for f in spam_features[:2]])
                    explanation += f"Although terms like {suspicious} appeared, their overall context remains benign."
                else:
                    explanation += "No suspicious phishing or spam trigger keywords were found."
            else:
                explanation = f"Classified as **SAFE / HAM** with **{conf}% confidence**. No spam triggers detected."

        return explanation

    def save(self, filepath):
        """Save model pipeline to disk"""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """Load trained model pipeline from disk"""
        return joblib.load(filepath)

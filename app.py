import os
from flask import Flask, render_template, request, jsonify, Response
from model import SpamClassifier
import database as db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'email-spam-classifier-secret-key-2026'

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
classifier = None

def get_or_load_model():
    global classifier
    if classifier is None:
        if os.path.exists(MODEL_FILE):
            print(f"Loading trained model from {MODEL_FILE}")
            classifier = SpamClassifier.load(MODEL_FILE)
        else:
            print("Model file not found! Training initial model...")
            import train_model
            train_model.train_and_evaluate()
            classifier = SpamClassifier.load(MODEL_FILE)
    return classifier

# Initialize DB on startup
with app.app_context():
    db.init_db()

# --- WEB PAGE ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

# --- API ENDPOINTS ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    subject = data.get('subject', '').strip()
    body = data.get('body', '').strip()
    threshold = float(data.get('threshold', 0.5))

    if not body:
        return jsonify({"error": "Email content cannot be empty"}), 400

    full_text = f"{subject}\n\n{body}".strip() if subject else body

    clf = get_or_load_model()
    pred_res = clf.predict(full_text, threshold=threshold)
    xai_res = clf.explain(full_text)

    # Save to history database
    record_id = db.save_prediction(
        subject=subject,
        full_text=full_text,
        prediction=pred_res["prediction"],
        confidence=pred_res["confidence"],
        spam_prob=pred_res["spam_probability"],
        ham_prob=pred_res["ham_probability"],
        risk_level=pred_res["risk_level"],
        trigger_words=xai_res["top_spam_words"],
        explanation_text=xai_res["explanation_text"]
    )

    response_payload = {
        "id": record_id,
        "prediction": pred_res["prediction"],
        "confidence": pred_res["confidence"],
        "spam_probability": pred_res["spam_probability"],
        "ham_probability": pred_res["ham_probability"],
        "risk_level": pred_res["risk_level"],
        "risk_badge": pred_res["risk_badge"],
        "threshold": threshold,
        "xai": {
            "top_spam_words": xai_res["top_spam_words"],
            "top_ham_words": xai_res["top_ham_words"],
            "highlighted_html": xai_res["highlighted_html"],
            "explanation_text": xai_res["explanation_text"],
            "total_significant_features": xai_res["total_significant_features"]
        }
    }
    return jsonify(response_payload)

@app.route('/api/history', methods=['GET'])
def api_history():
    search = request.args.get('search', '').strip() or None
    filter_type = request.args.get('filter', '').strip() or None
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    records = db.get_history(search=search, filter_type=filter_type, limit=limit, offset=offset)
    return jsonify({"history": records})

@app.route('/api/history/<int:record_id>', methods=['GET'])
def api_history_detail(record_id):
    item = db.get_prediction_by_id(record_id)
    if not item:
        return jsonify({"error": "Record not found"}), 404
    
    # Generate on-the-fly highlight if needed
    clf = get_or_load_model()
    xai_res = clf.explain(item["full_text"])
    item["highlighted_html"] = xai_res["highlighted_html"]
    return jsonify(item)

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def api_delete_history(record_id):
    db.delete_prediction(record_id)
    return jsonify({"success": True, "message": f"Record {record_id} deleted."})

@app.route('/api/history/<int:record_id>/feedback', methods=['POST'])
def api_feedback(record_id):
    data = request.get_json() or {}
    feedback = data.get('feedback', 'Reviewed')
    db.update_feedback(record_id, feedback)
    return jsonify({"success": True, "feedback": feedback})

@app.route('/api/admin/stats', methods=['GET'])
def api_admin_stats():
    stats = db.get_admin_dashboard_stats()
    clf = get_or_load_model()
    
    stats["model_info"] = {
        "algorithm": "TF-IDF + Logistic Regression (L2 Regularized)",
        "features_count": len(clf.feature_names_) if clf.feature_names_ is not None else 0,
        "n_gram_range": "1 to 2 (Unigrams & Bigrams)",
        "status": "Operational & Calibrated"
    }
    return jsonify(stats)

@app.route('/api/admin/clear', methods=['POST'])
def api_admin_clear():
    db.clear_all_history()
    return jsonify({"success": True, "message": "Prediction history has been reset successfully."})

@app.route('/api/export-csv', methods=['GET'])
def api_export_csv():
    csv_data = db.export_csv_data()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=spam_prediction_history.csv"}
    )

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    import sys
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

    # Ensure model is ready
    get_or_load_model()
    local_ip = get_local_ip()
    print("\n" + "="*65)
    print(" [*] Email Spam Classification Server Running!")
    print(f" [*] Local URL (Your PC):      http://127.0.0.1:5000")
    print(f" [*] Friends Link (Same Wi-Fi): http://{local_ip}:5000")
    print(" [*] Public Internet Link:      Run 'py share.py' in another terminal")
    print(" [*] Features: Confidence Score | Explainable AI | History | Admin")
    print("="*65 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

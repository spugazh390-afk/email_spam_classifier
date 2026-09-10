import sqlite3
import json
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam_history.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            subject TEXT,
            email_snippet TEXT,
            full_text TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            spam_probability REAL NOT NULL,
            ham_probability REAL NOT NULL,
            risk_level TEXT,
            trigger_words_json TEXT,
            explanation_text TEXT,
            feedback TEXT DEFAULT 'Unreviewed'
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(subject, full_text, prediction, confidence, spam_prob, ham_prob, risk_level, trigger_words, explanation_text):
    """
    Save a single classification record to SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    snippet = full_text.strip().replace("\n", " ")[:140]
    if len(full_text.strip()) > 140:
        snippet += "..."

    clean_subject = (subject or "No Subject").strip()
    trigger_json = json.dumps(trigger_words or [])

    cursor.execute("""
        INSERT INTO prediction_history (
            subject, email_snippet, full_text, prediction,
            confidence, spam_probability, ham_probability, risk_level,
            trigger_words_json, explanation_text, feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Unreviewed')
    """, (
        clean_subject, snippet, full_text, prediction,
        confidence, spam_prob, ham_prob, risk_level,
        trigger_json, explanation_text
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_history(search=None, filter_type=None, limit=50, offset=0):
    """
    Retrieve prediction history with optional search and filter.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM prediction_history WHERE 1=1"
    params = []

    if filter_type and filter_type.lower() in ['spam', 'ham']:
        query += " AND LOWER(prediction) = ?"
        params.append(filter_type.lower())

    if search:
        query += " AND (subject LIKE ? OR email_snippet LIKE ? OR full_text LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    results = []
    for r in rows:
        item = dict(r)
        try:
            item["trigger_words"] = json.loads(item.get("trigger_words_json") or "[]")
        except:
            item["trigger_words"] = []
        results.append(item)

    conn.close()
    return results

def get_prediction_by_id(record_id):
    """
    Retrieve full details for a single record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        item = dict(row)
        try:
            item["trigger_words"] = json.loads(item.get("trigger_words_json") or "[]")
        except:
            item["trigger_words"] = []
        return item
    return None

def delete_prediction(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='prediction_history'")
    conn.commit()
    conn.close()

def update_feedback(record_id, feedback):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE prediction_history SET feedback = ? WHERE id = ?", (feedback, record_id))
    conn.commit()
    conn.close()

def get_admin_dashboard_stats():
    """
    Aggregate statistics for the Admin Dashboard:
    - Total scans
    - Spam vs Ham counts and percentages
    - Average confidence
    - 7-day timeline trends
    - Most frequent spam keywords
    - Recent 5 scans
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total Scans
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    total_scans = cursor.fetchone()[0]

    # Spam Count
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE LOWER(prediction) = 'spam'")
    spam_count = cursor.fetchone()[0]

    # Ham Count
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE LOWER(prediction) = 'ham'")
    ham_count = cursor.fetchone()[0]

    # Average Confidence
    cursor.execute("SELECT AVG(confidence) FROM prediction_history")
    avg_conf_row = cursor.fetchone()[0]
    avg_confidence = round(avg_conf_row, 2) if avg_conf_row else 0.0

    spam_ratio = round((spam_count / total_scans * 100), 1) if total_scans > 0 else 0.0
    ham_ratio = round((ham_count / total_scans * 100), 1) if total_scans > 0 else 0.0

    # Scans over last 7 days
    timeline_labels = []
    timeline_spam = []
    timeline_ham = []

    for i in range(6, -1, -1):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        target_label = (datetime.now() - timedelta(days=i)).strftime('%b %d')
        timeline_labels.append(target_label)

        cursor.execute("""
            SELECT 
                SUM(CASE WHEN LOWER(prediction) = 'spam' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(prediction) = 'ham' THEN 1 ELSE 0 END)
            FROM prediction_history 
            WHERE DATE(timestamp) = DATE(?)
        """, (target_date,))
        spams_on_day, hams_on_day = cursor.fetchone()
        timeline_spam.append(spams_on_day or 0)
        timeline_ham.append(hams_on_day or 0)

    # Top spam trigger words across history
    cursor.execute("SELECT trigger_words_json FROM prediction_history WHERE LOWER(prediction) = 'spam'")
    spam_records = cursor.fetchall()
    keyword_freq = {}
    for r in spam_records:
        try:
            words = json.loads(r[0] or "[]")
            for w in words:
                word_str = w.get("word", "")
                if word_str:
                    keyword_freq[word_str] = keyword_freq.get(word_str, 0) + 1
        except:
            pass

    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    top_keywords_labels = [k[0] for k in sorted_keywords]
    top_keywords_counts = [k[1] for k in sorted_keywords]

    # Recent 6 scans
    cursor.execute("SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT 6")
    recent_rows = cursor.fetchall()
    recent_scans = [dict(r) for r in recent_rows]

    conn.close()

    return {
        "total_scans": total_scans,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_ratio": spam_ratio,
        "ham_ratio": ham_ratio,
        "avg_confidence": avg_confidence,
        "timeline": {
            "labels": timeline_labels,
            "spam": timeline_spam,
            "ham": timeline_ham
        },
        "top_keywords": {
            "labels": top_keywords_labels,
            "counts": top_keywords_counts
        },
        "recent_scans": recent_scans
    }

def export_csv_data():
    """
    Generate CSV string of all prediction history for admin export.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, subject, prediction, confidence, 
               spam_probability, ham_probability, risk_level, feedback, email_snippet
        FROM prediction_history
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Subject", "Prediction", "Confidence (%)", "Spam Prob (%)", "Ham Prob (%)", "Risk Level", "Feedback", "Snippet"])
    for row in rows:
        writer.writerow(list(row))
    return output.getvalue()

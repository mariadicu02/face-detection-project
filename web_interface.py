from flask import Flask, render_template, jsonify
import sqlite3
from datetime import datetime

DB_FILE = "prezenta.db"

app = Flask(__name__)

# ---------- helpers ----------
def db_connect():
    return sqlite3.connect(DB_FILE)

def get_active_session_id():
    """Returneaza ID-ul sesiunii active sau None."""
    conn = db_connect()
    c = conn.cursor()
    c.execute("SELECT id FROM sesiuni WHERE activa = 1 LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def get_attendance(limit=50):
    """Returneaza prezentele din sesiunea activa (timestamp, nume, metoda)."""
    session_id = get_active_session_id()
    if session_id is None:
        return [], 0                    

    conn = db_connect()
    c = conn.cursor()
    c.execute(
        '''
        SELECT p.timestamp,
               s.prenume || ' ' || s.nume AS nume_complet,
               p.metoda_detectie
        FROM prezente p
        JOIN studenti s ON p.student_id = s.id
        WHERE p.sesiune_id = ?
        ORDER BY p.timestamp DESC
        LIMIT ?
        ''',
        (session_id, limit)
    )
    prezente = c.fetchall()

    # total toate prezentele (nu doar limit)
    c.execute("SELECT COUNT(*) FROM prezente WHERE sesiune_id = ?", (session_id,))
    total = c.fetchone()[0]

    conn.close()
    return prezente, total

# ---------- routes ----------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/prezente')
def api_prezente():
    prezente, total = get_attendance()
    return jsonify({"total": total, "prezente": prezente})

# ---------- main ----------
if __name__ == '__main__':
    # ruleaza pe toate interfetele, port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)

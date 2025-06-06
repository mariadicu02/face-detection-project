from flask import Flask, render_template, jsonify
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

DATABASE = "prezenta.db"

def get_db_connection():
    """Obține conexiunea la baza de date"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Pagina principală"""
    return render_template('index.html')

@app.route('/api/session-info')
def get_session_info():
    """Returnează informații despre sesiunea activă"""
    conn = get_db_connection()
    session = conn.execute(
        'SELECT * FROM sesiuni WHERE activa = 1 LIMIT 1'
    ).fetchone()
    conn.close()
    
    if session:
        return jsonify({
            'id': session['id'],
            'nume': session['nume_sesiune'],
            'data_start': session['data_start'],
            'data_end': session['data_end'],
            'activa': bool(session['activa'])
        })
    else:
        return jsonify({'error': 'Nu există sesiune activă'})

@app.route('/api/students')
def get_students():
    """Returnează lista tuturor studenților"""
    conn = get_db_connection()
    students = conn.execute(
        'SELECT id, nume, prenume, email FROM studenti ORDER BY nume, prenume'
    ).fetchall()
    conn.close()
    
    return jsonify([{
        'id': student['id'],
        'nume': student['nume'],
        'prenume': student['prenume'],
        'email': student['email']
    } for student in students])

@app.route('/api/attendance')
def get_attendance():
    """Returnează prezențele pentru sesiunea activă"""
    conn = get_db_connection()
    
    # Găsește sesiunea activă
    session = conn.execute(
        'SELECT id FROM sesiuni WHERE activa = 1 LIMIT 1'
    ).fetchone()
    
    if not session:
        conn.close()
        return jsonify({'error': 'Nu există sesiune activă'})
    
    # Obține prezențele
    attendance = conn.execute('''
        SELECT p.id, p.student_id, p.metoda_detectie, p.timestamp,
               s.nume, s.prenume, s.email
        FROM prezente p
        JOIN studenti s ON p.student_id = s.id
        WHERE p.sesiune_id = ?
        ORDER BY p.timestamp DESC
    ''', (session['id'],)).fetchall()
    
    conn.close()
    
    return jsonify([{
        'id': row['id'],
        'student_id': row['student_id'],
        'nume_complet': f"{row['prenume']} {row['nume']}",
        'email': row['email'],
        'metoda_detectie': row['metoda_detectie'],
        'timestamp': row['timestamp']
    } for row in attendance])

@app.route('/api/statistics')
def get_statistics():
    """Returnează statistici despre prezență"""
    conn = get_db_connection()
    
    # Găsește sesiunea activă
    session = conn.execute(
        'SELECT id FROM sesiuni WHERE activa = 1 LIMIT 1'
    ).fetchone()
    
    if not session:
        conn.close()
        return jsonify({'error': 'Nu există sesiune activă'})
    
    # Total studenți înrolați
    total_students = conn.execute('SELECT COUNT(*) FROM studenti').fetchone()[0]
    
    # Studenți prezenți în sesiunea activă
    present_students = conn.execute(
        'SELECT COUNT(DISTINCT student_id) FROM prezente WHERE sesiune_id = ?',
        (session['id'],)
    ).fetchone()[0]
    
    # Prezențe pe metode de detectie
    methods = conn.execute('''
        SELECT metoda_detectie, COUNT(*) as count
        FROM prezente 
        WHERE sesiune_id = ?
        GROUP BY metoda_detectie
    ''', (session['id'],)).fetchall()
    
    conn.close()
    
    method_stats = {row['metoda_detectie']: row['count'] for row in methods}
    
    return jsonify({
        'total_studenti': total_students,
        'studenti_prezenti': present_students,
        'procent_prezenta': round((present_students / total_students * 100) if total_students > 0 else 0, 1),
        'metode_detectie': method_stats
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
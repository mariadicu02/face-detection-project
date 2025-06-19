# creeaza_db.py
import sqlite3
from datetime import datetime

conn = sqlite3.connect('prezenta.db')
c = conn.cursor()

# Creează tabelele
c.execute('''
CREATE TABLE IF NOT EXISTS studenti (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nume TEXT NOT NULL,
    prenume TEXT NOT NULL,
    email TEXT NOT NULL
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS sesiuni (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nume_sesiune TEXT NOT NULL,
    data_start TEXT NOT NULL,
    data_end TEXT NOT NULL,
    activa INTEGER DEFAULT 0
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS prezente (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    sesiune_id INTEGER NOT NULL,
    metoda_detectie TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES studenti(id),
    FOREIGN KEY(sesiune_id) REFERENCES sesiuni(id)
)
''')

# Adaugă date de test
c.execute("INSERT INTO studenti (nume, prenume, email) VALUES ('Popescu', 'Ana', 'ana.popescu@example.com')")
c.execute("INSERT INTO studenti (nume, prenume, email) VALUES ('Ionescu', 'Mihai', 'mihai.ionescu@example.com')")

c.execute("INSERT INTO sesiuni (nume_sesiune, data_start, data_end, activa) VALUES (?, ?, ?, 1)",
          ('Sesiune Iunie', datetime.now().isoformat(), datetime.now().isoformat()))

# Asociază prezențe
c.execute("INSERT INTO prezente (student_id, sesiune_id, metoda_detectie, timestamp) VALUES (1, 1, 'Facial', ?)",
          (datetime.now().isoformat(),))
c.execute("INSERT INTO prezente (student_id, sesiune_id, metoda_detectie, timestamp) VALUES (2, 1, 'NFC', ?)",
          (datetime.now().isoformat(),))

conn.commit()
conn.close()
print("Baza de date `prezenta.db` a fost creată cu succes.")

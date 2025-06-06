#!/usr/bin/env python3
"""
Script de configurare și management pentru sistemul de prezență
"""

import os
import sys
import sqlite3
import json
import subprocess
from pathlib import Path

def check_requirements():
    """Verifică dacă toate dependențele sunt instalate"""
    required_packages = [
        'opencv-python',
        'dlib',
        'numpy',
        'requests',
        'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Lipsesc următoarele pachete Python:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstalează-le cu:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Toate pachetele Python sunt instalate")
    return True

def check_dlib_models():
    """Verifică dacă modelele dlib sunt prezente"""
    required_models = [
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    missing_models = []
    for model in required_models:
        if not os.path.exists(model):
            missing_models.append(model)
    
    if missing_models:
        print("❌ Lipsesc următoarele modele dlib:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nDescarcă-le de la:")
        print("- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("- http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        return False
    
    print("✅ Toate modelele dlib sunt prezente")
    return True

def create_directories():
    """Creează directoarele necesare"""
    directories = [
        "poze_moodle",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Director creat/verificat: {directory}")

def create_flask_template():
    """Creează template-ul Flask"""
    template_content = """<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Prezență - Monitorizare Live</title>
    <!-- CSS content would be here - use the HTML artifact content -->
</head>
<body>
    <!-- HTML content would be here - use the HTML artifact content -->
</body>
</html>"""
    
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(template_content)
    
    print("ℹ️  Template Flask creat. Copiază conținutul din artifact-ul HTML în templates/index.html")

def setup_database():
    """Configurează baza de date inițială"""
    from attendance_system import init_database
    
    try:
        init_database()
        print("✅ Baza de date configurată cu succes")
    except Exception as e:
        print(f"❌ Eroare la configurarea bazei de date: {e}")

def show_configuration_info():
    """Afișează informații de configurare"""
    print("\n" + "="*60)
    print("📋 INFORMAȚII DE CONFIGURARE")
    print("="*60)
    print("\n1. CONFIGURARE MOODLE:")
    print("   - URL Moodle: Modifică MOODLE_URL în scriptul principal")
    print("   - Token Moodle: Setează variabila de mediu MOODLE_TOKEN")
    print("   - Course ID: Modifică COURSE_ID în scriptul principal")
    
    print("\n2. CONFIGURARE NFC (opțional):")
    print("   - Instalează: pip install adafruit-circuitpython-pn532")
    print("   - Conectează modulul PN532 la pinii I2C ai Raspberry Pi")
    
    print("\n3. PORNIREA SISTEMULUI:")
    print("   - Script principal: python attendance_system.py")
    print("   - Interfață web: python web_interface.py")
    print("   - Accesează: http://localhost:5000 sau http://IP_RASPBERRY:5000")
    
    print("\n4. TESTARE:")
    print("   - Pune poze de test în directorul poze_moodle/")
    print("   - Verifică că camera funcționează")
    print("   - Testează conexiunea la Moodle")

def create_systemd_service():
    """Creează un serviciu systemd pentru pornirea automată"""
    current_dir = os.getcwd()
    service_content = f"""[Unit]
Description=Attendance System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={current_dir}
ExecStart={sys.executable} {current_dir}/attendance_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "/etc/systemd/system/attendance-system.service"
    
    try:
        with open("attendance-system.service", "w") as f:
            f.write(service_content)
        
        print("✅ Fișier serviciu creat: attendance-system.service")
        print("Pentru a-l instala:")
        print(f"sudo cp attendance-system.service {service_file}")
        print("sudo systemctl enable attendance-system.service")
        print("sudo systemctl start attendance-system.service")
        
    except Exception as e:
        print(f"❌ Eroare la crearea serviciului: {e}")

def show_database_stats():
    """Afișează statistici din baza de date"""
    try:
        conn = sqlite3.connect("prezenta.db")
        c = conn.cursor()
        
        # Numărul de studenți
        c.execute("SELECT COUNT(*) FROM studenti")
        num_students = c.fetchone()[0]
        
        # Numărul de sesiuni
        c.execute("SELECT COUNT(*) FROM sesiuni")
        num_sessions = c.fetchone()[0]
        
        # Numărul de prezențe
        c.execute("SELECT COUNT(*) FROM prezente")
        num_attendance = c.fetchone()[0]
        
        print(f"\n📊 STATISTICI BAZA DE DATE:")
        print(f"   - Studenți înregistrați: {num_students}")
        print(f"   - Sesiuni create: {num_sessions}")
        print(f"   - Prezențe înregistrate: {num_attendance}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Nu se poate citi baza de date: {e}")

def add_nfc_uid():
    """Adaugă UID NFC pentru un student"""
    try:
        conn = sqlite3.connect("prezenta.db")
        c = conn.cursor()
        
        # Afișează studenții disponibili
        c.execute("SELECT id, nume, prenume FROM studenti ORDER BY nume, prenume")
        students = c.fetchall()
        
        if not students:
            print("❌ Nu există studenți în baza de date")
            return
        
        print("\n👥 STUDENȚI DISPONIBILI:")
        for i, (student_id, nume, prenume) in enumerate(students, 1):
            print(f"   {i}. {prenume} {nume} (ID: {student_id})")
        
        choice = input("\nSelectează numărul studentului: ")
        try:
            student_index = int(choice) - 1
            if 0 <= student_index < len(students):
                student_id, nume, prenume = students[student_index]
                
                nfc_uid = input(f"Introdu UID NFC pentru {prenume} {nume}: ").strip().upper()
                
                if nfc_uid:
                    c.execute("UPDATE studenti SET nfc_uid = ? WHERE id = ?", (nfc_uid, student_id))
                    conn.commit()
                    print(f"✅ UID NFC adăugat pentru {prenume} {nume}")
                else:
                    print("❌ UID invalid")
            else:
                print("❌ Selecție invalidă")
        except ValueError:
            print("❌ Introdu un număr valid")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Eroare: {e}")

def main():
    """Funcția principală"""
    print("🎓 CONFIGURARE SISTEM PREZENȚĂ")
    print("="*40)
    
    while True:
        print("\nSelectează o opțiune:")
        print("1. Verifică cerințele sistemului")
        print("2. Configurare inițială completă")
        print("3. Creează directoarele necesare")
        print("4. Configurează baza de date")
        print("5. Creează serviciu systemd")
        print("6. Afișează statistici baza de date")
        print("7. Adaugă UID NFC pentru student")
        print("8. Afișează informații configurare")
        print("0. Ieșire")
        
        choice = input("\nOpțiunea ta: ").strip()
        
        if choice == "1":
            check_requirements()
            check_dlib_models()
        elif choice == "2":
            print("\n🚀 CONFIGURARE COMPLETĂ...")
            if check_requirements() and check_dlib_models():
                create_directories()
                create_flask_template()
                setup_database()
                create_systemd_service()
                show_configuration_info()
        elif choice == "3":
            create_directories()
        elif choice == "4":
            setup_database()
        elif choice == "5":
            create_systemd_service()
        elif choice == "6":
            show_database_stats()
        elif choice == "7":
            add_nfc_uid()
        elif choice == "8":
            show_configuration_info()
        elif choice == "0":
            print("👋 La revedere!")
            break
        else:
            print("❌ Opțiune invalidă")

if __name__ == "__main__":
    main()
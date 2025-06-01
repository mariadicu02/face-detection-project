import requests

# === CONFIGURARE ===
MOODLE_URL = "http://192.168.0.101/moodle/webservice/rest/server.php"
MOODLE_TOKEN = "5a987e634310317873b8831b2faeaded"
MOODLE_FUNCTION = "core_webservice_get_site_info"

# === Construim URL-ul complet ===
url = f"{MOODLE_URL}?wstoken={MOODLE_TOKEN}&wsfunction={MOODLE_FUNCTION}&moodlewsrestformat=json"

try:
    print(f"[INFO] Trimit cerere la: {url}")
    response = requests.get(url, timeout=5)
    print(f"[INFO] Status HTTP: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            print("[✅] Conexiune reușită. Token-ul este valid.")
            print("[INFO] Răspuns Moodle:")
            print(data)
        except ValueError:
            print("[❌] Eroare: Serverul nu a trimis un răspuns JSON valid.")
            print(f"Conținut brut: {response.text}")
    else:
        print(f"[❌] Eroare HTTP: {response.status_code}")
except Exception as e:
    print(f"[❌] Eroare la conectare: {e}")

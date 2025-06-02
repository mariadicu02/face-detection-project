import board
import busio
import adafruit_pn532.i2c

# Inițializare I2C
i2c = busio.I2C(board.SCL, board.SDA)
pn532 = adafruit_pn532.i2c.PN532_I2C(i2c, debug=False)

# Inițializare PN532
pn532.SAM_configuration()

print("Apropie un card NFC...")

while True:
    uid = pn532.read_passive_target(timeout=0.5)
    if uid:
        print(f"Card detectat! UID: {uid.hex().upper()}")

from py532lib.i2c import *
from py532lib.frame import *
from py532lib.constants import *


nfc = Pn532_i2c()
nfc.SAMconfigure()

print("Apropie un card MIFARE de cititor...")

try:
    while True:
        uid = nfc.read_mifare().get_data()
        if uid:
            print("Card detectat! UID: {}".format(uid.hex()))
except KeyboardInterrupt:
    print("\nScanare oprită de utilizator.")

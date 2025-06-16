from RPLCD.i2c import CharLCD

# Inițializează LCD-ul cu adresa ta: 0x27
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, charmap='A00', auto_linebreaks=True)

lcd.clear()
lcd.write_string("Salut, Razvan!\nRegleaza LCD-ul")
# Nu se face sleep, nu se dă clear!

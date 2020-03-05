# Cheltuieli lunare
cheltuieli = {
    'chirie': 566,
    'digi': 20,
    'gaz': 70,
    'enel': 20,
    'intretinere': 70,
    'telefon': 35,
    'sala': 135,
    'tuns': 70,
    'rata': 183,
    'metrou': 35,
    'extra': 100,
    'mancare': 750}

suma_cheltuieli = sum(cheltuieli.values())
# Venituri lunare
income_10 = 2600 - suma_cheltuieli
income_25 = 5000
# Sold conturi
BCR = 12000
ING = 300
crestere_dupa_1_an = 2500
for year in [2020, 2021]:
    luni_trecute = ['January', 'February']
    for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        if year == 2020 and month in luni_trecute:
            continue
        else:
            BCR += income_25
            ING += income_10
            print('_' * 40)
            print(f'Balance in 10th of {month}, {year}:')
            print(f'BCR balance: {BCR} RON or {round(BCR/4.6)}\u20ac')
            print(f'BRD balance: {round(ING)} RON or {round(ING / 4.6)}\u20ac')
            print(f'Total balance: {round(BCR + ING)} RON or {round((ING + BCR) / 4.6)}\u20ac')
    income_25 += crestere_dupa_1_an
print('_' * 40)
print(f'Salariu este: +{income_10 + income_25 + suma_cheltuieli} RON')
print(f'Cheltuielile totale sunt: -{suma_cheltuieli} RON')

# Cheltuieli mobila
dormitor = {
    'draperii': 170,
    'dressing': 3000,
    'pat': 1500,
    'saltea': 1300,
    'pilota': 150,
    'perna': 100 * 2,
    'lenjerie': 100 * 2,
    'cearsaf': 60 * 2,
    'noptiera': 130,
    'covor': 100
}

bucatarie = {
    'masa': 170,
    'ansamblu': 3000,
    'scaun': 140 * 4,
    'frigider': 1000,
    'cuptor': 600,
    'plita': 600,
    'chiuveta': 1000,
    'centrala': 0,
    'covor': 60
}

sufragerie = {
    'canapea': 3000,
    'ansamblu': 1500,
    'masuta': 200,
    'covor': 300,
    'dulap': 500,
    'fotoliu': 0,
    'veioza': 100,
    'aspirator': 300,
    'televizor': 0
}

hol = {
    'dressing': 1000,
    'cuier': 200,
    'masina_spalat': 1200,
    'covor': 50
}

baie = {
    'corp': 500,
    'chiuveta': 500,
    'cada': 1000,
    'cos_rufe': 50,
    'covoras': 30
}

total_mobila = sum(dormitor.values()) + sum(bucatarie.values()) + sum(sufragerie.values()) + sum(hol.values()) + sum(baie.values())
print('_'*40)
print(f'Total mobila : {total_mobila} RON or {total_mobila / 4.6}\u20ac')

# Cheltuielile cu apartamentul
credit_apartament = 70000 # €
durata_credit = 360
coeficient_inapoiere = 1.8
rata_apartament = credit_apartament * coeficient_inapoiere / durata_credit

credit_ikea = 3000
rata_ikea = credit_ikea / 60

credit_emag = 2000
rata_emag = credit_emag / 36

avans = credit_apartament * 0.05
acte = credit_apartament * 0.015

investitie_initiala = avans + acte + total_mobila / 4.6

print('_'*40)
print('Rate:')
print(f'apartament: {rata_apartament * 4.6} RON or {rata_apartament}\u20ac')
print(f'ikea : {rata_ikea * 4.6} RON or {rata_ikea}\u20ac')
print(f'emag : {rata_emag * 4.6} RON or {rata_emag}\u20ac')
print(f'Total rate : {rata_emag * 4.6 + rata_emag * 4.6 + rata_apartament * 4.6} RON or {rata_emag + rata_ikea + rata_apartament}\u20ac')
print('investitie_initiala:')
print(f'avans : {avans * 4.6} RON or {avans}\u20ac')
print(f'acte : {acte * 4.6} RON or {acte}\u20ac')
print(f'mobila : {total_mobila} RON or {total_mobila / 4.6}\u20ac')
print(f'Total : {avans * 4.6 + acte * 4.6} RON or {avans + acte}\u20ac')
print('_'*40)
print(f'Bilant lunar. ')
print(f'Salariu lunar: +{income_10 + income_25 + suma_cheltuieli} RON')
print(f'Cheltuieli lunare: -{suma_cheltuieli - cheltuieli["chirie"]} RON -{rata_emag * 4.6 + rata_emag * 4.6 + rata_apartament * 4.6} RON = -{suma_cheltuieli - cheltuieli["chirie"] + rata_emag * 4.6 + rata_emag * 4.6 + rata_apartament * 4.6}')
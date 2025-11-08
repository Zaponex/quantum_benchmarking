# Hilfsfunktionen zum Dekodieren
def invert_map(d):
    return {idx: key for key, idx in d.items()}
    # idx -> (tname, rname, z)

def decode_sample(sample_dict, inv_x, inv_y, inv_w):
    chosen_x = []
    chosen_y = []
    chosen_w = []
    for idx, val in sample_dict.items():
        if val != 1: 
            continue
        if idx in inv_x:
            chosen_x.append(('x',) + inv_x[idx])
        elif idx in inv_y:
            chosen_y.append(('y',) + inv_y[idx])
        elif idx in inv_w:
            chosen_w.append(('w',) + inv_w[idx])
        else:
            # ggf. weitere Variabelfamilien hier ergänzen
            pass
    return chosen_x, chosen_y, chosen_w

# Beispielaufruf (NumPy-Typen zu ints konvertieren, falls nötig):


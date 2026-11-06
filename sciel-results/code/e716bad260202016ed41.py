def construct():
    moduli = [2, 4, 8, 16, 32, 64, 128]
    residues = [0, 1, 3, 7, 15, 31, 63]
    return [[r, m] for r, m in zip(residues, moduli)]

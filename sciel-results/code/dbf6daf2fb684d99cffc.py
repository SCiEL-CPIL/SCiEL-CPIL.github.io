def construct():
    # Use K=11
    K = 11
    moduli = [K * i for i in range(1, 12)] # 11, 22, ..., 121
    # All are <= 128
    # Residues 0..10
    result = []
    for idx, m in enumerate(moduli):
        r = idx
        if r < m:
            result.append([r, m])
    return result

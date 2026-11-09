DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    STRASSEN_TERMS = (
        (1, 0, 0, 1, 1, 0, 0, 1),
        (0, 0, 1, 1, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 1, 0, -1),
        (0, 0, 0, 1, -1, 0, 1, 0),
        (1, 1, 0, 0, 0, 0, 0, 1),
        (-1, 0, 1, 0, 1, 1, 0, 0),
        (0, 1, 0, -1, 0, 0, 1, 1),
    )
    
    STRASSEN_RECON = (
        (1, 0, 0, 1, -1, 0, 1),
        (0, 0, 1, 0, 1, 0, 0),
        (0, 1, 0, 1, 0, 0, 0),
        (1, -1, 1, 0, 0, 1, 0),
    )
    
    TRIVIAL_TERMS = (
        (1, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 1, 0),
        (0, 0, 1, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0),
    )
    
    TRIVIAL_RECON = (
        (1, 1, 0, 0),
        (0, 0, 1, 1),
    )
    
    def populate_full2x2x2(io, jo, ko, base):
        for t in range(7):
            c = base + t
            x00, x01, x10, x11, y00, y01, y10, y11 = STRASSEN_TERMS[t]
            r00, r01, r10, r11 = (STRASSEN_RECON[z][t] for z in range(4))
            
            a00 = io * m + jo
            a01 = io * m + jo + 1
            a10 = (io + 1) * m + jo
            a11 = (io + 1) * m + jo + 1
            
            b00 = jo * p + ko
            b01 = jo * p + ko + 1
            b10 = (jo + 1) * p + ko
            b11 = (jo + 1) * p + ko + 1
            
            c00 = ko * n + io
            c01 = (ko + 1) * n + io
            c10 = ko * n + (io + 1)
            c11 = (ko + 1) * n + (io + 1)
            
            if x00: U[a00][c] += x00
            if x01: U[a01][c] += x01
            if x10: U[a10][c] += x10
            if x11: U[a11][c] += x11
            if y00: V[b00][c] += y00
            if y01: V[b01][c] += y01
            if y10: V[b10][c] += y10
            if y11: V[b11][c] += y11
            if r00: W[c00][c] += r00
            if r01: W[c01][c] += r01
            if r10: W[c10][c] += r10
            if r11: W[c11][c] += r11
    
    def populate_2x2x1(io, jo, ko, base):
        for t in range(4):
            c = base + t
            x00, x01, x10, x11, y00, y01, y10, y11 = TRIVIAL_TERMS[t]
            r00, r10 = TRIVIAL_RECON[0][t], TRIVIAL_RECON[1][t]
            
            a00 = io * m + jo
            a01 = io * m + jo + 1
            a10 = (io + 1) * m + jo
            a11 = (io + 1) * m + jo + 1
            
            b00 = jo * p + ko
            b10 = (jo + 1) * p + ko
            
            c00 = ko * n + io
            c10 = ko * n + (io + 1)
            
            if x00: U[a00][c] += x00
            if x01: U[a01][c] += x01
            if x10: U[a10][c] += x10
            if x11: U[a11][c] += x11
            if y00: V[b00][c] += y00
            if y10: V[b10][c] += y10
            if r00: W[c00][c] += r00
            if r10: W[c10][c] += r10
    
    populate_full2x2x2(0, 0, 0, 0)
    populate_full2x2x2(0, 0, 2, 7)
    populate_full2x2x2(0, 0, 4, 14)
    populate_2x2x1(0, 0, 6, 21)
    
    populate_full2x2x2(0, 2, 0, 25)
    populate_full2x2x2(0, 2, 2, 32)
    populate_full2x2x2(0, 2, 4, 39)
    populate_2x2x1(0, 2, 6, 46)
    
    return {"U": U, "V": V, "W": W}

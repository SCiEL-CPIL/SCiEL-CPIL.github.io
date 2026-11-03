import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        # M1 = (A11+A22)(B11+B22)
        add_term({ar0*4+ac0: 1, ar1*4+ac1: 1}, {br0*6+bc0: 1, br1*6+bc1: 1}, {cc0*3+cr0: 1, cc1*3+cr1: 1})
        # M2 = (A21+A22)B11
        add_term({ar1*4+ac0: 1, ar1*4+ac1: 1}, {br0*6+bc0: 1}, {cc0*3+cr1: 1})
        # M3 = A11(B12-B22)
        add_term({ar0*4+ac0: 1}, {br0*6+bc1: 1, br1*6+bc1: -1}, {cc1*3+cr0: 1, cc1*3+cr1: 1})
        # M4 = A22(B21-B11)
        add_term({ar1*4+ac1: 1}, {br1*6+bc0: 1, br0*6+bc0: -1}, {cc0*3+cr0: 1, cc0*3+cr1: 1})
        # M5 = (A11+A12)B22
        add_term({ar0*4+ac0: 1, ar0*4+ac1: 1}, {br1*6+bc1: 1}, {cc0*3+cr0: -1, cc1*3+cr0: 1})
        # M6 = (A21-A11)(B11+B12)
        add_term({ar1*4+ac0: 1, ar0*4+ac0: -1}, {br0*6+bc0: 1, br0*6+bc1: 1}, {cc1*3+cr1: 1})
        # M7 = (A12-A22)(B21+B22)
        add_term({ar0*4+ac1: 1, ar1*4+ac1: -1}, {br1*6+bc0: 1, br1*6+bc1: 1}, {cc0*3+cr0: 1})
    
    def direct_1x2x2x2(ar, ac0, ac1, br0, br1, bc0, bc1, cr, cc0, cc1):
        add_term({ar*4+ac0: 1}, {br0*6+bc0: 1}, {cc0*3+cr: 1})
        add_term({ar*4+ac0: 1}, {br1*6+bc0: 1}, {cc0*3+cr: 1})
        add_term({ar*4+ac1: 1}, {br0*6+bc1: 1}, {cc1*3+cr: 1})
        add_term({ar*4+ac1: 1}, {br1*6+bc1: 1}, {cc1*3+cr: 1})
    
    # C[0:2, 0:2]
    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    strassen_2x2(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    # C[0:2, 2:4]
    strassen_2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    strassen_2x2(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    # C[0:2, 4:6]
    strassen_2x2(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    strassen_2x2(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)
    # C[2:3, 0:2]
    direct_1x2x2x2(2, 0, 1, 0, 1, 0, 1, 2, 0, 1)
    direct_1x2x2x2(2, 2, 3, 2, 3, 0, 1, 2, 0, 1)
    # C[2:3, 2:4]
    direct_1x2x2x2(2, 0, 1, 0, 1, 2, 3, 2, 2, 3)
    direct_1x2x2x2(2, 2, 3, 2, 3, 2, 3, 2, 2, 3)
    # C[2:3, 4:6]
    direct_1x2x2x2(2, 0, 1, 0, 1, 4, 5, 2, 4, 5)
    direct_1x2x2x2(2, 2, 3, 2, 3, 4, 5, 2, 4, 5)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(18)]
    
    for c, (u, v, w) in enumerate(terms):
        for row, val in u.items():
            U[row][c] = val
        for row, val in v.items():
            V[row][c] = val
        for row, val in w.items():
            W[row][c] = val
    
    return {"U": U, "V": V, "W": W}

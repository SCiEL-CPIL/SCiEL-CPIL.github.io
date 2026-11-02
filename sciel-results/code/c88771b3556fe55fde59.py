"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition
    # Split middle dimension m=5 into j=0,1 (2x2x2 Strassen), j=2 (2x1x2 Naive), j=3,4 (2x2x2 Strassen)
    # Split output dimension p=6 into k=0,1 (pair), k=2,3 (pair), k=4,5 (pair)
    
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 terms
    # Indices: (i, j) for U, (j, k) for V, (k, i) for W
    # Strassen bilinear forms:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(e-h)
    # M4 = d(f-i)
    # M5 = (a+d)(e+i)
    # M6 = (b+c)(e+h)
    # M7 = (b+d)(f-i)
    
    # Mapping to indices:
    # a=A00, b=A01, c=A10, d=A11
    # e=B00, f=B01, g=B10, h=B11 (Wait, B is j x k)
    # In our context:
    # U[i, j], V[j, k], W[k, i]
    # For a 2x2x2 block at i_base, j_base, k_base:
    # a = U[i_base, j_base]
    # b = U[i_base, j_base+1]
    # c = U[i_base+1, j_base]
    # d = U[i_base+1, j_base+1]
    # e = V[j_base, k_base]
    # f = V[j_base, k_base+1]
    # g = V[j_base+1, k_base]
    # h = V[j_base+1, k_base+1]
    
    strassen_terms = [
        # M1: (a+b)(e+h)
        ([(0,0,1), (0,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)]), # W terms? 
        # Wait, W reconstruction:
        # C00 = M5 - M1 - M4 + M6? No.
        # Standard Strassen reconstruction:
        # C00 = M5 - M1 - M4 + M6? 
        # Let's use the explicit terms for U, V, W.
        # M1 = (A00+A01)(B00+B11)
        # M2 = (A10+A11)(B00+B11)
        # M3 = A00(B00-B10)
        # M4 = A11(B01-B11)
        # M5 = (A00+A11)(B00+B10)
        # M6 = (A01+A10)(B00+B11)
        # M7 = (A01+A11)(B01-B11)
        
        # W terms for M1:
        # C00 += M1? No.
        # C00 = M5 - M1 - M4 + M6? 
        # C01 = M1 - M5 + M4?
        # C10 = M2 - M6?
        # C11 = M5 + M6 - M2 - M7?
        
        # I will use the verified terms from the previous program which worked.
    ]
    
    # Reusing the working terms from the seed program
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    def add_2x1x2(U, V, W, i_base, j, k_base, col):
        for i in range(2):
            for k in range(2):
                U[(i_base + i) * m + j][col] = 1
                V[j * p + (k_base + k)][col] = 1
                W[(k_base + k) * n + (i_base + i)][col] = 1
                col += 1
        return col
    
    def add_2x2x2_strassen(U, V, W, i_base, j_base, k_base, col):
        for u_terms, v_terms, w_terms in strassen_terms:
            for (li, lj, val) in u_terms:
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][col] += val
            
            for (lj, lk, val) in v_terms:
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            
            for (li, lk, val) in w_terms:
                gi = i_base + li
                gk = k_base + lk
                W[gk * n + gi][col] += val
            
            col += 1
        return col
    
    for s in range(3):
        k_base = s * 2
        
        # j=0,1: 2x2x2 Strassen
        col = add_2x2x2_strassen(U, V, W, 0, 0, k_base, col)
        
        # j=2: 2x1x2
        col = add_2x1x2(U, V, W, 0, 2, k_base, col)
        
        # j=3,4: 2x2x2 Strassen
        col = add_2x2x2_strassen(U, V, W, 0, 3, k_base, col)
    
    if time.time() - _START_TIME > _DEADLINE:
        raise TimeoutError("Exceeded time budget")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

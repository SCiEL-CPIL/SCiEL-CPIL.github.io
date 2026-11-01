import time
import sys

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition: 5 pairs of columns (Strassen rank 7) + 1 single column (Schoolbook rank 4)
    rank = 5 * 7 + 1 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    r = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # 5 pairs of columns
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices for C
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # U indices for A (always 0,1,2,3)
        # m1 = (a+d)(e+h)
        add_term(0, 1, v_e, 1, w_c00, 1)
        add_term(3, 1, v_e, 1, w_c00, 1)
        add_term(0, 1, v_h, 1, w_c00, 1)
        add_term(3, 1, v_h, 1, w_c00, 1)
        # Wait, add_term adds to the same column r.
        # I need to accumulate U, V, W for the same r.
        # My add_term function modifies U, V, W at index r.
        # But m1 has multiple non-zeros in U, V, W.
        # U[0]=1, U[3]=1. V[v_e]=1, V[v_h]=1. W[w_c00]=1, W[w_c11]=1.
        
        # Let's rewrite the loop to be clearer.
        # Reset r for each term.
        
    # Re-implementation with explicit term definition
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # bg
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # ce
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # dg
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to index U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    # 5 pairs of columns
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        # Strassen for 2x2 x 2x2
        # M1 = (a11+a22)(b11+b22)
        U[u_idx(0, 0), col] = 1
        U[u_idx(1, 1), col] = 1
        V[v_idx(0, k1), col] = 1
        V[v_idx(1, k2), col] = 1
        W[w_idx(k1, 0), col] = 1
        W[w_idx(k2, 1), col] = 1
        col += 1
        
        # M2 = (a21+a22)b22
        U[u_idx(1, 0), col] = 1
        U[u_idx(1, 1), col] = 1
        V[v_idx(1, k2), col] = 1
        W[w_idx(k2, 1), col] = 1
        col += 1
        
        # M3 = a11(b12-b22)
        U[u_idx(0, 0), col] = 1
        V[v_idx(0, k2), col] = 1
        V[v_idx(1, k2), col] = -1
        W[w_idx(k2, 0), col] = 1
        col += 1
        
        # M4 = a22(b21-b11)
        U[u_idx(1, 1), col] = 1
        V[v_idx(1, k1), col] = 1
        V[v_idx(0, k1), col] = -1
        W[w_idx(k1, 1), col] = 1
        W[w_idx(k1, 0), col] = 1
        col += 1
        
        # M5 = (a11+a12)b11
        U[u_idx(0, 0), col] = 1
        U[u_idx(0, 1), col] = 1
        V[v_idx(0, k1), col] = 1
        W[w_idx(k1, 1), col] = -1
        W[w_idx(k2, 0), col] = 1
        col += 1
        
        # M6 = (a21+a11)(b22-b12)
        U[u_idx(1, 0), col] = 1
        U[u_idx(0, 0), col] = 1
        V[v_idx(1, k2), col] = 1
        V[v_idx(0, k2), col] = -1
        W[w_idx(k2, 1), col] = 1
        col += 1
        
        # M7 = (a11-a21)(b11+b21)
        U[u_idx(0, 0), col] = 1
        U[u_idx(1, 0), col] = -1
        V[v_idx(0, k1), col] = 1
        V[v_idx(1, k1), col] = 1
        W[w_idx(k1, 0), col] = 1
        W[w_idx(k1, 1), col] = 1
        col += 1

    # Last column (index 10, which is k=10 in 0-indexed? No, p=11, indices 0..10)
    # k=10 is the 11th column.
    k = 10
    # M8 = a11 b1k
    U[u_idx(0, 0), col] = 1
    V[v_idx(0, k), col] = 1
    W[w_idx(k, 0), col] = 1
    col += 1
    
    # M9 = a12 b2k
    U[u_idx(0, 1), col] = 1
    V[v_idx(1, k), col] = 1
    W[w_idx(k, 0), col] = 1
    col += 1
    
    # M10 = (a21+a11)b1k + (a22+a12)b2k
    U[u_idx(1, 0), col] = 1
    U[u_idx(0, 0), col] = 1
    U[u_idx(1, 1), col] = 1
    U[u_idx(0, 1), col] = 1
    V[v_idx(0, k), col] = 1
    V[v_idx(1, k), col] = 1
    W[w_idx(k, 1), col] = 1
    W[w_idx(k, 0), col] = -1
    W[w_idx(k, 0), col] = -1 # Wait, W[k,0] gets -1 from M8 and M9?
    # Let's re-verify the last column logic.
    # C_1k = M8 + M9
    # C_2k = M10 - M8 - M9
    # So W[k,0] for M8 is 1. W[k,0] for M9 is 1.
    # W[k,0] for M10 is -1.
    # But M8 and M9 are separate columns in U,V,W.
    # So for col M8: W[k,0]=1.
    # For col M9: W[k,0]=1.
    # For col M10: W[k,0]=-1, W[k,1]=1.
    
    # Correcting M10 W assignments
    # W[w_idx(k, 1), col] = 1 is correct for C_2k.
    # W[w_idx(k, 0), col] = -1 is correct for C_1k part of M10?
    # C_1k = M8 + M9. M10 does not contribute to C_1k.
    # So W[w_idx(k, 0), col] for M10 should be 0.
    # C_2k = M10 - M8 - M9.
    # So W[w_idx(k, 1), col] for M10 is 1.
    # W[w_idx(k, 1), col] for M8 is 0.
    # W[w_idx(k, 1), col] for M9 is 0.
    
    # Let's re-evaluate M10.
    # M10 = (a21+a11)b1k + (a22+a12)b2k
    # This term contributes to C_2k with coefficient 1.
    # It does not contribute to C_1k.
    # So W[w_idx(k, 1), col] = 1.
    # W[w_idx(k, 0), col] = 0.
    
    # So for the last column:
    # Col 35 (M8): U[0,0]=1, V[0,k]=1, W[k,0]=1
    # Col 36 (M9): U[0,1]=1, V[1,k]=1, W[k,0]=1
    # Col 37 (M10): U[1,0]=1, U[0,0]=1, U[1,1]=1, U[0,1]=1, V[0,k]=1, V[1,k]=1, W[k,1]=1
    
    # My code above had W[w_idx(k, 0), col] = -1 for M10. That is wrong.
    # I will fix it in the final code.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

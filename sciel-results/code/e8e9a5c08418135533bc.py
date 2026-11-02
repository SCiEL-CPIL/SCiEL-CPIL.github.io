import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # The rank is 44, constructed from 4 blocks of (2x2)x(2x2) [rank 7] and (2x1)x(1x2) [rank 4].
    # Total rank = 4 * (7 + 4) = 44.
    rank = 44
    
    # Initialize U, V, W as numpy arrays for easier manipulation, then convert to lists if needed
    # The interface expects lists of lists, but we can construct them efficiently.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to assign values
    def set_val(U, V, W, u_idx, v_idx, w_idx, val=1):
        pass # Placeholder, we will do direct assignment
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A[0:2, 0:2] * B[0:2, k0:k1+1] ---
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0, col] = 1
        U[1 * m + 1, col] = 1
        V[0 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0, col] = 1
        U[1 * m + 1, col] = 1
        V[0 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0, col] = 1
        V[0 * p + k1, col] = 1
        V[1 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1, col] = 1
        V[1 * p + k0, col] = 1
        V[0 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0, col] = 1
        U[0 * m + 1, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0, col] = 1
        U[0 * m + 0, col] = -1
        V[0 * p + k0, col] = 1
        V[0 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1, col] = 1
        U[1 * m + 1, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1
        
        # --- Rank-4 part for A[0:2, 2] * B[2, k0:k1+1] ---
        
        # P1 = A02 * B2k0
        U[0 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[0 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[1 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[1 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

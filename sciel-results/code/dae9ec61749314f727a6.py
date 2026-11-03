"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 64 decomposition
    rank = 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen M_2,2,2 block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices i in {0,1}, j in {0,1}, k in {0,1}
        # Global i = i_off + i, j = j_off + j, k = k_off + k
        # U index: (i_off+i)*m + (j_off+j)
        # V index: (j_off+j)*p + (k_off+k)
        # W index: (k_off+k)*n + (i_off+i)
        
        # Strassen 7 terms
        # M1 = A00 * (B00 + B11)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # M2 = (A00 + A01) * B11
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # M3 = (A10 + A11) * B00
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # M5 = (A00 + A11) * (B00 + B11)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M6 = (A01 - A11) * (B10 + B11)
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # M7 = (A00 - A10) * (B00 + B01)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # Helper to add M_1,4,4 block (rank 4)
    def add_m144(i_idx, k_off):
        nonlocal col
        # i is fixed to i_idx
        # j in 0..3, k in k_off..k_off+3
        for r in range(4):
            # U[i_idx, j=r] = 1
            U[i_idx * m + r, col] = 1
            # V[j=r, k] = 1 for all k? No, V[j,k] is 1 if j==r?
            # My derivation: U_{0,j,r} = delta_{j,r}, V_{j,k,r} = delta_{j,r}, W_{k,0,r} = 1
            # So V[j,k,r] is 1 if j==r, else 0.
            V[r * p + (k_off + 0), col] = 1 # Wait, V index is j*p + k
            # V[j, k] = delta_{j,r}. So V[r, k] = 1 for all k?
            # No, V_{j,k,r} = delta_{j,r}.
            # So V[r, k_off+0, col] = 1
            # V[r, k_off+1, col] = 1
            # V[r, k_off+2, col] = 1
            # V[r, k_off+3, col] = 1
            for k_local in range(4):
                V[r * p + (k_off + k_local), col] = 1
            
            # W[k, i_idx] = 1 for all k
            for k_local in range(4):
                W[(k_off + k_local) * n + i_idx, col] = 1
            
            col += 1

    # Construct M_3,4,4 for k in 0..3
    # i in 0..1: M_2,4,4
    # j in 0..1: M_2,2,4 -> 2x M_2,2,2
    add_strassen(0, 0, 0) # i=0,1 j=0,1 k=0,1
    add_strassen(0, 0, 2) # i=0,1 j=0,1 k=2,3
    # j in 2..3: M_2,2,4 -> 2x M_2,2,2
    add_strassen(0, 2, 0) # i=0,1 j=2,3 k=0,1
    add_strassen(0, 2, 2) # i=0,1 j=2,3 k=2,3
    
    # i in 2..2: M_1,4,4
    add_m144(2, 0)
    
    # Construct M_3,4,4 for k in 4..7
    # i in 0..1: M_2,4,4
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_strassen(0, 2, 4)
    add_strassen(0, 2, 6)
    
    # i in 2..2: M_1,4,4
    add_m144(2, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

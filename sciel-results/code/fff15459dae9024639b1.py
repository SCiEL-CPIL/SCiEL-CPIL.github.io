import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
        return False
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    return True

def construct():
    U = np.zeros((n*m, 34))
    V = np.zeros((m*p, 34))
    W = np.zeros((p*n, 34))
    
    def add_strassen_block(ib, jb, kb, col_offset):
        # M1
        U[ib*m+jb, col_offset] = 1; U[(ib+1)*m+jb+1, col_offset] = 1
        V[jb*p+kb, col_offset] = 1; V[(jb+1)*p+kb+1, col_offset] = 1
        W[kb*n+ib, col_offset] = 1; W[(kb+1)*n+ib+1, col_offset] = 1
        W[kb*n+ib+1, col_offset] = -1; W[(kb+1)*n+ib, col_offset] = -1
        
        # M2
        U[(ib+1)*m+jb, col_offset+1] = 1; U[(ib+1)*m+jb+1, col_offset+1] = 1
        V[jb*p+kb, col_offset+1] = 1
        W[kb*n+ib+1, col_offset+1] = 1
        
        # M3
        U[ib*m+jb, col_offset+2] = 1
        V[(jb+1)*p+kb, col_offset+2] = 1; V[jb*p+kb, col_offset+2] = -1
        W[kb*n+ib, col_offset+2] = 1
        
        # M4
        U[(ib+1)*m+jb+1, col_offset+3] = 1
        V[jb*p+kb+1, col_offset+3] = 1; V[(jb+1)*p+kb+1, col_offset+3] = -1
        W[(kb+1)*n+ib+1, col_offset+3] = 1
        
        # M5
        U[ib*m+jb, col_offset+4] = 1; U[ib*m+jb+1, col_offset+4] = 1
        V[(jb+1)*p+kb+1, col_offset+4] = 1
        W[(kb+1)*n+ib, col_offset+4] = 1
        
        # M6
        U[(ib+1)*m+jb, col_offset+5] = 1; U[ib*m+jb, col_offset+5] = -1
        V[jb*p+kb, col_offset+5] = 1; V[jb*p+kb+1, col_offset+5] = 1
        W[kb*n+ib+1, col_offset+5] = 1; W[(kb+1)*n+ib+1, col_offset+5] = 1
        
        # M7
        U[ib*m+jb+1, col_offset+6] = 1; U[(ib+1)*m+jb+1, col_offset+6] = -1
        V[(jb+1)*p+kb, col_offset+6] = 1; V[(jb+1)*p+kb+1, col_offset+6] = 1
        W[kb*n+ib, col_offset+6] = 1; W[(kb+1)*n+ib, col_offset+6] = 1
        
    # Block 1: rows 0,1 (i), cols 0,1 (j), cols 0,1 (k)
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: rows 1,2 (i), cols 0,1 (j), cols 2,3 (k)
    # Covers i=1,2; j=0,1; k=2,3
    # Remaining for j=2: (0,2,0), (0,2,1), (1,2,2), (1,2,3), (2,2,2), (2,2,3)
    # Wait, Block 1 covers (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # Block 2 (1,0,2) covers (1,0,2),(1,0,3),(1,1,2),(1,1,3),(2,0,2),(2,0,3),(2,1,2),(2,1,3)
    # Total covered:
    # i=0: j=0(k0,1), j=1(k0,1) -> Missing j=2(k0,1,2,3) -> 4 terms
    # i=1: j=0(k0,1,2,3), j=1(k0,1,2,3) -> Missing j=2(k0,1,2,3) -> 4 terms
    # i=2: j=0(k2,3), j=1(k2,3) -> Missing j=2(k0,1,2,3) -> 4 terms
    # Total remaining: 12 terms.
    # 14 + 12 = 26. This is better than 34!
    
    # Let's verify this tiling works.
    # Block 1: (0,0,0)
    # Block 2: (1,0,2)
    # 
    # Direct terms for j=2:
    # (0,2,0), (0,2,1), (0,2,2), (0,2,3)
    # (1,2,0), (1,2,1), (1,2,2), (1,2,3)
    # (2,2,0), (2,2,1), (2,2,2), (2,2,3)
    # All are disjoint from the blocks?
    # Block 1 covers j=0,1. Block 2 covers j=0,1.
    # So yes, j=2 is completely uncovered by blocks.
    # This gives 14 + 12 = 26.
    
    # Can we do better?
    # What if we use a block for j=2?
    # Block 3: (0,1,0) -> Covers i=0,1; j=1,2; k=0,1
    # Overlaps with Block 1 at j=1.
    # If we use Block 1 (0,0,0) and Block 3 (0,1,0)
    # Covered:
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 3: i=0,1; j=1,2; k=0,1
    # Union: i=0,1; j=0,1,2; k=0,1
    # Missing:
    # i=0,1: j=0,1,2; k=2,3 -> 6 terms
    # i=2: j=0,1,2; k=0,1,2,3 -> 12 terms
    # Total missing: 18.
    # Total rank: 14 + 18 = 32. Worse.
    
    # What if we use Block 1 (0,0,0) and Block 4 (0,2,0)?
    # Block 4 covers i=0,1; j=2 (only, since m=3, j+1=3 out of bounds? No, j ranges 0..m-1.
    # My _strassen_2x2_block assumes 2x2 blocks.
    # If j=2, j+1=3 which is out of bounds for m=3.
    # So we can only place blocks at j=0,1.
    # Similarly for i=0,1 and k=0,1,2.
    
    # So the best tiling with 2x2x2 Strassen blocks seems to be:
    # Block 1: (0,0,0)
    # Block 2: (1,0,2)
    # Rank 26.
    
    # Let's add the direct terms.
    col = 14
    for i in range(3):
        for k in range(4):
            j = 2
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
            
    # Total rank 26.
    
    # Can we get 25 or less?
    # Maybe by overlapping blocks with coefficients?
    # Or a different decomposition?
    # For now, return 26.
    
    U = U[:, :26]
    V = V[:, :26]
    W = W[:, :26]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Rank:", U.shape[1])
    print("Valid:", _verify(U, V, W))


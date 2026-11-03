import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            I = (i0, i1)
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    J = (j0, j1)
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            K = (k0, k1)
                            # Store block and its terms
                            terms = set()
                            for i in I:
                                for j in J:
                                    for k in K:
                                        terms.add((i, j, k))
                            blocks.append((I, J, K, terms))
                            
    # Greedy search for max disjoint blocks
    best_blocks = []
    best_count = 0
    
    # Randomized restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random.shuffle(blocks)
        covered = set()
        current_blocks = []
        
        for block in blocks:
            I, J, K, terms = block
            if not (terms & covered):
                covered |= terms
                current_blocks.append(block)
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
        # Optional: Local search to improve?
        # For now, just restarts.
        
    # Construct U, V, W
    num_blocks = len(best_blocks)
    remaining_terms = n * m * p - num_blocks * 8
    rank = num_blocks * 7 + remaining_terms
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    covered = set()
    
    # Strassen coefficients for 2x2x2 block
    # Local indices 0, 1
    # U_t[i,j], V_t[j,k], W_t[k,i]
    # t=0..6
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2: (A10+A11)B00 -> C10+=1, C11-=1
    # M3: A00(B01-B11) -> C01+=1, C11+=1
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    # M6: (A10-A00)(B00+B01) -> C11+=1
    # M7: (A01-A11)(B10+B11) -> C00+=1
    
    # Note: W_t[k,i] is coeff of M_t in C_{i,k}
    
    # Precompute Strassen vectors
    # U: 4x7, V: 4x7, W: 4x7 (local indices flattened)
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1 # C00, C11
    
    # M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1 # C10 (k=0, i=1 -> idx 1? No. k*n+i. n=2 local. k=0,i=1 -> 1. k=1,i=1 -> 3)
    # Wait, W index is k*n+i.
    # C10: i=1, k=0. Index 0*2+1 = 1.
    # C11: i=1, k=1. Index 1*2+1 = 3.
    W_strassen[1, 1] = 1
    W_strassen[3, 1] = -1
    
    # M3
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    # C01: i=0, k=1. Index 1*2+0 = 2.
    # C11: i=1, k=1. Index 3.
    W_strassen[2, 2] = 1
    W_strassen[3, 2] = 1
    
    # M4
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    # C00: i=0, k=0. Index 0.
    # C10: i=1, k=0. Index 1.
    W_strassen[0, 3] = 1
    W_strassen[1, 3] = 1
    
    # M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    # C00: i=0, k=0. Index 0. Coeff -1.
    # C01: i=0, k=1. Index 2. Coeff 1.
    W_strassen[0, 4] = -1
    W_strassen[2, 4] = 1
    
    # M6
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    # C11: i=1, k=1. Index 3. Coeff 1.
    W_strassen[3, 5] = 1
    
    # M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    # C00: i=0, k=0. Index 0. Coeff 1.
    W_strassen[0, 6] = 1
    
    # Add Strassen blocks
    for I, J, K, terms in best_blocks:
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        # Mapping local (0,1) to global
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        for t in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    val = U_strassen[li*2+lj, t]
                    if val != 0:
                        gi = i0 if li == 0 else i1
                        gj = j0 if lj == 0 else j1
                        U[gi*m + gj, col] += val
                        
            # V
            for lj in range(2):
                for lk in range(2):
                    val = V_strassen[lj*2+lk, t]
                    if val != 0:
                        gj = j0 if lj == 0 else j1
                        gk = k0 if lk == 0 else k1
                        V[gj*p + gk, col] += val
                        
            # W
            for lk in range(2):
                for li in range(2):
                    val = W_strassen[lk*2+li, t]
                    if val != 0:
                        gk = k0 if lk == 0 else k1
                        gi = i0 if li == 0 else i1
                        W[gk*n + gi, col] += val
                        
            col += 1
        covered |= terms
        
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    # Trim
    if col < U.shape[1]:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
    # Verify
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term to the matrices
    # u_indices: list of (idx, val) for U
    # v_indices: list of (idx, val) for V
    # w_indices: list of (idx, val) for W
    def add_term(r, u_idx_val, v_idx_val, w_idx_val):
        for idx, val in u_idx_val:
            U[idx, r] += val
        for idx, val in v_idx_val:
            V[idx, r] += val
        for idx, val in w_idx_val:
            W[idx, r] += val

    r = 0
    
    # Define Strassen terms for a 2x2x2 block
    # Local indices i,j in {0,1}, k in {0,1}
    # Global indices:
    # U[i, j] -> i*m + j  (but here m is local size 2 for the block, offset by j_block_start)
    # V[j, k] -> j*p + k  (j_block_start + j) * p + (k_block_start + k)
    # W[k, i] -> (k_block_start + k) * n + i
    
    def get_strassen_terms(j_start, k_start):
        terms = []
        
        # 1. (a+d)(e+h)
        # a=A00, d=A11, e=B00, h=B11
        u = [((0)*2 + 0, 1), ((1)*2 + 1, 1)]
        v = [((0)*8 + 0, 1), ((1)*8 + 1, 1)]
        w = [(0*2 + 0, 1), (1*2 + 1, 1)]
        terms.append((u, v, w))
        
        # 2. (c+d)e
        # c=A10, d=A11, e=B00
        u = [((1)*2 + 0, 1), ((1)*2 + 1, 1)]
        v = [((0)*8 + 0, 1)]
        w = [(0*2 + 1, 1), (1*2 + 1, -1)]
        terms.append((u, v, w))
        
        # 3. a(f-h)
        # a=A00, f=B01, h=B11
        u = [((0)*2 + 0, 1)]
        v = [((0)*8 + 1, 1), ((1)*8 + 1, -1)]
        w = [(1*2 + 0, 1), (1*2 + 1, 1)]
        terms.append((u, v, w))
        
        # 4. d(g-e)
        # d=A11, g=B10, e=B00
        u = [((1)*2 + 1, 1)]
        v = [((1)*8 + 0, 1), ((0)*8 + 0, -1)]
        w = [(0*2 + 0, 1), (0*2 + 1, 1)]
        terms.append((u, v, w))
        
        # 5. (a+b)h
        # a=A00, b=A01, h=B11
        u = [((0)*2 + 0, 1), ((0)*2 + 1, 1)]
        v = [((1)*8 + 1, 1)]
        w = [(0*2 + 0, -1), (1*2 + 0, 1)]
        terms.append((u, v, w))
        
        # 6. (c-a)(e+f)
        # c=A10, a=A00, e=B00, f=B01
        u = [((1)*2 + 0, 1), ((0)*2 + 0, -1)]
        v = [((0)*8 + 0, 1), ((0)*8 + 1, 1)]
        w = [(1*2 + 1, 1)]
        terms.append((u, v, w))
        
        # 7. (b-d)(g+h)
        # b=A01, d=A11, g=B10, h=B11
        u = [((0)*2 + 1, 1), ((1)*2 + 1, -1)]
        v = [((1)*8 + 0, 1), ((1)*8 + 1, 1)]
        w = [(0*2 + 0, 1)]
        terms.append((u, v, w))
        
        return terms

    # Apply Strassen blocks
    # Block 1: j_start=0, k_start=0
    for u, v, w in get_strassen_terms(0, 0):
        # Map local indices to global
        # U local (i, j) -> global (i, j_start + j)
        # V local (j, k) -> global (j_start + j, k_start + k)
        # W local (k, i) -> global (k_start + k, i)
        
        # We need to adjust the indices inside the lists
        # The lists above are hardcoded for j_start=0, k_start=0.
        # So for other blocks, we must shift indices.
        pass

    # Let's refine the helper to take offsets
    def add_strassen_block(j_off, k_off, r_start):
        # 1
        add_term(r_start + 0, 
                 [(j_off, 1), (j_off + 3, 1)], 
                 [(k_off, 1), (k_off + 3, 1)], 
                 [(k_off, 1), (k_off + 3, 1)]) # W idx: k*n+i. (0,0)->0, (1,1)->1*2+1=3. Wait.
        
        # Let's re-verify W indices.
        # W index = k * n + i.
        # n=2.
        # (k=0, i=0) -> 0
        # (k=0, i=1) -> 1
        # (k=1, i=0) -> 2
        # (k=1, i=1) -> 3
        
        # Term 1: (a+d)(e+h)
        # C00 += A00B00 + A00B11 + A11B00 + A11B11
        # C11 += A11B11 + A11B00 + A00B11 + A00B00
        # C01 += A00B00 + A00B11 + A11B00 + A11B11 ? No.
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M3 + M4 - M5
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 - M6 + M7
        
        # Let's map these to W coefficients for a single block.
        # The W vector for this block will be added to the global W.
        # Since the blocks are disjoint in terms of K indices (mostly), we can just add them.
        # Wait, the K indices are disjoint for the 4 blocks.
        # Block 1: k=0,1. Block 2: k=2,3. etc.
        # So W indices for Block 1 are 0,1,2,3 (k=0,1, i=0,1).
        # W indices for Block 2 are 4,5,6,7 (k=2,3, i=0,1).
        
        # So we can define the W coefficients for a generic 2x2x2 block relative to local k=0,1.
        
        return r_start + 7

    # Reset r
    r = 0
    
    # Block 1: j=0,1; k=0,1
    # We will just hardcode the 28 Strassen terms with correct global indices.
    
    def apply_strassen(j_off, k_off, r_cur):
        # 1. (A00+A11)(B00+B11)
        U[j_off, r_cur] += 1       # i=0, j=j_off
        U[j_off + 3, r_cur] += 1   # i=1, j=j_off+1 -> idx = 1*2 + 1 = 3? No.
        
        # Index calculation check:
        # U index = i * m + j.
        # m=3.
        # Block j_off=0.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        
        # Block j_off=1? No, the blocks are on K, not J.
        # The J dimension is always 0 and 1 for the Strassen part.
        # So j is always 0 and 1.
        # U indices are fixed: 0, 1, 3, 4.
        
        # V index = j * p + k.
        # p=8.
        # Block k_off=0.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 8
        # j=1, k=1 -> 9
        
        # W index = k * n + i.
        # n=2.
        # Block k_off=0.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # Block k_off=2.
        # k=2, i=0 -> 4
        # k=2, i=1 -> 5
        # k=3, i=0 -> 6
        # k=3, i=1 -> 7
        
        # So, for a block at k_off, the W indices are k_off, k_off+1, k_off+2, k_off+3.
        # The V indices are (k_off, k_off+1) for j=0, and (8+k_off, 8+k_off+1) for j=1.
        
        # U indices are always 0, 1, 3, 4 (for i=0,1 and j=0,1).
        
        # Let's define the local W coefficients for C00, C01, C10, C11 in terms of M1..M7.
        # M1: C00(+1), C11(+1)
        # M2: C00(-1), C10(+1)
        # M3: C00(+1), C11(-1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(-1)
        # M7: C01(+1), C11(+1)
        
        # W_local[0] (C00) = 1 - 1 + 1 + 1 - 1 = 1
        # W_local[1] (C01) = 0 + 0 + 0 + 0 + 1 + 0 + 1 = 2 ? No.
        
        # Let's re-read Strassen formulas carefully.
        # C00 = M1 - M2 + M3 + M4 - M5
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 - M6 + M7
        
        # So:
        # W[M1]: C00=1, C01=0, C10=0, C11=1
        # W[M2]: C00=-1, C01=0, C10=1, C11=0
        # W[M3]: C00=1, C01=0, C10=0, C11=-1
        # W[M4]: C00=1, C01=0, C10=1, C11=0
        # W[M5]: C00=-1, C01=1, C10=0, C11=0
        # W[M6]: C00=0, C01=0, C10=0, C11=-1
        # W[M7]: C00=0, C01=1, C10=0, C11=1
        
        # U coeffs:
        # M1: A00(1), A11(1) -> U[0]=1, U[4]=1
        # M2: A10(1), A11(1) -> U[3]=1, U[4]=1
        # M3: A00(1) -> U[0]=1
        # M4: A11(1) -> U[4]=1
        # M5: A00(1), A01(1) -> U[0]=1, U[1]=1
        # M6: A10(1), A00(-1) -> U[3]=1, U[0]=-1
        # M7: A01(1), A11(-1) -> U[1]=1, U[4]=-1
        
        # V coeffs:
        # M1: B00(1), B11(1) -> V[0]=1, V[9]=1 (for k_off=0)
        # M2: B00(1) -> V[0]=1
        # M3: B01(1), B11(-1) -> V[1]=1, V[9]=-1
        # M4: B10(1), B00(-1) -> V[8]=1, V[0]=-1
        # M5: B11(1) -> V[9]=1
        # M6: B00(1), B01(1) -> V[0]=1, V[1]=1
        # M7: B10(1), B11(1) -> V[8]=1, V[9]=1
        
        # Generalize for k_off:
        # V[0] -> V[k_off]
        # V[1] -> V[k_off+1]
        # V[8] -> V[8+k_off]
        # V[9] -> V[9+k_off]
        
        # W indices for k_off:
        # C00 -> W[k_off]
        # C01 -> W[k_off+1]
        # C10 -> W[k_off+2]
        # C11 -> W[k_off+3]
        
        # Term 1
        U[0, r_cur] += 1; U[4, r_cur] += 1
        V[k_off, r_cur] += 1; V[9+k_off, r_cur] += 1
        W[k_off, r_cur] += 1; W[k_off+3, r_cur] += 1
        r_cur += 1
        
        # Term 2
        U[3, r_cur] += 1; U[4, r_cur] += 1
        V[k_off, r_cur] += 1
        W[k_off, r_cur] += -1; W[k_off+2, r_cur] += 1
        r_cur += 1
        
        # Term 3
        U[0, r_cur] += 1
        V[k_off+1, r_cur] += 1; V[9+k_off, r_cur] += -1
        W[k_off, r_cur] += 1; W[k_off+3, r_cur] += -1
        r_cur += 1
        
        # Term 4
        U[4, r_cur] += 1
        V[8+k_off, r_cur] += 1; V[k_off, r_cur] += -1
        W[k_off, r_cur] += 1; W[k_off+2, r_cur] += 1
        r_cur += 1
        
        # Term 5
        U[0, r_cur] += 1; U[1, r_cur] += 1
        V[9+k_off, r_cur] += 1
        W[k_off, r_cur] += -1; W[k_off+1, r_cur] += 1
        r_cur += 1
        
        # Term 6
        U[3, r_cur] += 1; U[0, r_cur] += -1
        V[k_off, r_cur] += 1; V[k_off+1, r_cur] += 1
        W[k_off+3, r_cur] += -1
        r_cur += 1
        
        # Term 7
        U[1, r_cur] += 1; U[4, r_cur] += -1
        V[8+k_off, r_cur] += 1; V[9+k_off, r_cur] += 1
        W[k_off+1, r_cur] += 1; W[k_off+3, r_cur] += 1
        r_cur += 1
        
        return r_cur

    r = 0
    r = apply_strassen(0, 0, r)
    r = apply_strassen(0, 2, r)
    r = apply_strassen(0, 4, r)
    r = apply_strassen(0, 6, r)
    
    # Schoolbook for j=2
    # m=1 part.
    # U indices: i*m + j = i*3 + 2.
    # i=0 -> 2
    # i=1 -> 5
    # V indices: j*p + k = 2*8 + k = 16 + k.
    # W indices: k*n + i.
    
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            r += 1
            
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


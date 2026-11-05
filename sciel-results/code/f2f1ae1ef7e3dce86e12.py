"""Combined program: Block Strassen with optimized remainders."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify exact decomposition including zero constraints."""
    nm, mp, pn = n * m, m * p, p * n
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for (a, b, c) in valid:
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s - 1.0) > tol:
            return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We will build the decomposition manually.
    # Rank estimate: 40.
    # Let's allocate a bit more for safety, say 50.
    max_rank = 50
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    
    current_rank = 0
    
    def add_factor(u_idx, v_idx, w_idx, val=1.0):
        nonlocal current_rank
        if current_rank >= max_rank:
            return
        U[u_idx, current_rank] += val
        V[v_idx, current_rank] += val
        W[w_idx, current_rank] += val
        current_rank += 1

    # Strassen factors for 2x2x2
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for a 2x2x2 block:
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    def add_strassen_block(i0, j0, k0):
        # M1
        # (A[i0, j0] + A[i0+1, j0+1]) * (B[j0, k0] + B[j0+1, k0+1])
        # This is not a single term, it's a product of sums.
        # In tensor decomposition, we need to expand this.
        # Wait, Strassen is a decomposition of the tensor itself.
        # The tensor for 2x2x2 has rank 7.
        # The 7 terms are the M's.
        # But the M's are products of sums of A and B entries.
        # This means the factors U, V, W will have non-zero entries in multiple positions.
        
        # Let's define the 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
        # T = sum_r U_r V_r W_r
        
        # r=0: M1
        # U: A11, A22 are 1.
        # V: B11, B22 are 1.
        # W: C11, C22 are 1.
        # Check: A11 B11 + A11 B22 + A22 B11 + A22 B22.
        # C11 gets A11 B11 + A11 B22 + A22 B11 + A22 B22? No.
        # Strassen formula:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map M's to the output C.
        # Each M contributes to specific C entries with coefficients +/-1.
        
        # M1: (A11+A22)(B11+B22)
        # U: [1, 0, 1, 0] (A11, A12, A21, A22) -> 1 at 0, 2
        # V: [1, 0, 0, 1] (B11, B12, B21, B22) -> 1 at 0, 3
        # W: C11 gets +1, C22 gets +1.
        
        # Let's hardcode the 7 patterns.
        # Local indices:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # M1: (A00+A11)(B00+B11)
        # U: 1 at 00, 11
        # V: 1 at 00, 11
        # W: +1 at C00, +1 at C11
        
        # M2: (A10+A11)B00
        # U: 1 at 10, 11
        # V: 1 at 00
        # W: +1 at C10, -1 at C11
        
        # M3: A00(B10-B11)
        # U: 1 at 00
        # V: 1 at 10, -1 at 11
        # W: +1 at C01, +1 at C11
        
        # M4: A11(B01-B00)
        # U: 1 at 11
        # V: 1 at 01, -1 at 00
        # W: +1 at C00, +1 at C10
        
        # M5: (A00+A01)B11
        # U: 1 at 00, 01
        # V: 1 at 11
        # W: -1 at C00, +1 at C01
        
        # M6: (A10-A00)(B00+B01)
        # U: 1 at 10, -1 at 00
        # V: 1 at 00, 01
        # W: +1 at C11
        
        # M7: (A01-A11)(B10+B11)
        # U: 1 at 01, -1 at 11
        # V: 1 at 10, 11
        # W: +1 at C00
        
        # We need to add these to the global U, V, W.
        
        a_idx = lambda i, j: i * m + j
        b_idx = lambda j, k: j * p + k
        c_idx = lambda k, i: k * n + i
        
        # M1
        r = current_rank; current_rank += 1
        U[a_idx(i0, j0), r] = 1
        U[a_idx(i0+1, j0+1), r] = 1
        V[b_idx(j0, k0), r] = 1
        V[b_idx(j0+1, k0+1), r] = 1
        W[c_idx(k0, i0), r] = 1
        W[c_idx(k0+1, i0+1), r] = 1
        
        # M2
        r = current_rank; current_rank += 1
        U[a_idx(i0+1, j0), r] = 1
        U[a_idx(i0+1, j0+1), r] = 1
        V[b_idx(j0, k0), r] = 1
        W[c_idx(k0, i0+1), r] = 1
        W[c_idx(k0+1, i0+1), r] = -1
        
        # M3
        r = current_rank; current_rank += 1
        U[a_idx(i0, j0), r] = 1
        V[b_idx(j0+1, k0), r] = 1
        V[b_idx(j0+1, k0+1), r] = -1
        W[c_idx(k0+1, i0), r] = 1
        W[c_idx(k0+1, i0+1), r] = 1
        
        # M4
        r = current_rank; current_rank += 1
        U[a_idx(i0+1, j0+1), r] = 1
        V[b_idx(j0, k0+1), r] = 1
        V[b_idx(j0, k0), r] = -1
        W[c_idx(k0, i0), r] = 1
        W[c_idx(k0, i0+1), r] = 1
        
        # M5
        r = current_rank; current_rank += 1
        U[a_idx(i0, j0), r] = 1
        U[a_idx(i0, j0+1), r] = 1
        V[b_idx(j0+1, k0+1), r] = 1
        W[c_idx(k0, i0), r] = -1
        W[c_idx(k0+1, i0), r] = 1
        
        # M6
        r = current_rank; current_rank += 1
        U[a_idx(i0+1, j0), r] = 1
        U[a_idx(i0, j0), r] = -1
        V[b_idx(j0, k0), r] = 1
        V[b_idx(j0, k0+1), r] = 1
        W[c_idx(k0+1, i0+1), r] = 1
        
        # M7
        r = current_rank; current_rank += 1
        U[a_idx(i0, j0+1), r] = 1
        U[a_idx(i0+1, j0+1), r] = -1
        V[b_idx(j0+1, k0), r] = 1
        V[b_idx(j0+1, k0+1), r] = 1
        W[c_idx(k0, i0), r] = 1

    # Add 4 Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 4)
    
    # Add remainder for i=2
    # i=2, j in 0..3, k in 0..6
    # Rank 4 decomposition for 1x4x7
    # C_k = sum_j A_j B_jk
    # We can use 4 terms:
    # r=0: A_0 B_0k -> C_k gets A_0 B_0k? No.
    # We need C_k = A_0 B_0k + A_1 B_1k + A_2 B_2k + A_3 B_3k.
    # Term 0: U[2m+0]=1, V[0p+k]=1, W[kn+2]=1.
    # This gives A_0 B_0k to C_k.
    # We need to sum over j.
    # So for each j, we add a term.
    # But V depends on k.
    # For j=0: V[0p+k]=1 for all k.
    # This is one rank 1 tensor?
    # U[2m+0]=1. V[0p+k]=1 (for all k). W[kn+2]=1 (for all k).
    # This term contributes A_0 B_0k to C_k.
    # Yes.
    # So we need 4 terms, one for each j.
    
    for j in range(4):
        r = current_rank; current_rank += 1
        U[2*m + j, r] = 1
        # V[j*p + k, r] = 1 for all k
        for k in range(p):
            V[j*p + k, r] = 1
        # W[k*n + 2, r] = 1 for all k
        for k in range(p):
            W[k*n + 2, r] = 1

    # Add remainder for i in 0,1, k=6
    # i in 0,1. j in 0..3. k=6.
    # C_i6 = sum_j A_ij B_j6.
    # For i=0: C_06 = A_00 B_06 + A_01 B_16 + A_02 B_26 + A_03 B_36.
    # For i=1: C_16 = A_10 B_06 + A_11 B_16 + A_12 B_26 + A_13 B_36.
    # These are independent.
    # For i=0, j=0: Term U[0m+0]=1, V[0p+6]=1, W[6n+0]=1.
    # This gives A_00 B_06 to C_06.
    # We need 4 terms for i=0 (j=0..3) and 4 terms for i=1 (j=0..3).
    
    for i in [0, 1]:
        for j in range(4):
            r = current_rank; current_rank += 1
            U[i*m + j, r] = 1
            V[j*p + 6, r] = 1
            W[6*n + i, r] = 1

    # Final rank
    final_rank = current_rank
    U = U[:, :final_rank]
    V = V[:, :final_rank]
    W = W[:, :final_rank]
    
    # Verify
    if _verify(U, V, W, n, m, p, final_rank):
        def _to_list(arr):
            return [[float(x) for x in row] for row in np.asarray(arr)]
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sch = np.zeros((nm, rank_school))
        V_sch = np.zeros((mp, rank_school))
        W_sch = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i * m + j, col] = 1.0
                    V_sch[j * p + k, col] = 1.0
                    W_sch[k * n + i, col] = 1.0
                    col += 1
        def _to_list(arr):
            return [[float(x) for x in row] for row in np.asarray(arr)]
        return {"U": _to_list(U_sch), "V": _to_list(V_sch), "W": _to_list(W_sch)}

# EVOLVE-BLOCK-END

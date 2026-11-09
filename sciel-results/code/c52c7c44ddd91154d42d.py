"""Matrix multiplication 3x5x5: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """
    Verify exact decomposition using numpy with Fraction objects.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k'*n+i',c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    We reorganize: for each c, define matrices:
      A_c: n x m, A_c[i,j] = U[i*m+j, c]
      B_c: m x p, B_c[j,k] = V[j*p+k, c]
      C_c: p x n, C_c[k,i] = W[k*n+i, c]
    
    Then the tensor is sum_c A_c ⊗ B_c ⊗ C_c (outer products).
    The multiplication tensor T has T[i,j,k; i',j',k'] = delta_{j,j'} (wait, let me be careful)
    
    Actually for matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    The tensor is T_{i,j,k} = A_{i,j} B_{j,k} -> (AB)_{i,k}
    
    In our indexing:
    U[i*m+j, c] -> A_c[i,j]
    V[j*p+k, c] -> B_c[j,k]  
    W[k*n+i, c] -> C_c[k,i]
    
    The condition is: sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k'*n+i',c] = 1 if i==i', j==j', k==k' else 0
    
    Wait, that's not right. Let me re-read the convention.
    
    "output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed as W[k*n+i, r]. This means for the output (AB)_{i,k}, we use W[k*n+i, r].
    
    The tensor decomposition condition for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    In tensor form: T_{i,j,k} such that C = sum_{j} A_{i,j} B_{j,k}
    
    The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
    U_c has entries U_c[i,j] = U[i*m+j, c]  (size n x m)
    V_c has entries V_c[j,k] = V[j*p+k, c]  (size m x p)  
    W_c has entries W_c[k,i] = W[k*n+i, c]  (size p x n)
    
    And the condition is:
    sum_c U_c[i,j] * V_c[j',k] * W_c[k,i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k} ... 
    
    Hmm, actually the standard way: the matrix multiplication tensor M ∈ R^{n x m x p} is defined by
    M[i,j,k] = 1 if we think of it as the structure tensor.
    
    Let me just use the verification formula from the seed:
    For all i,j,k,i',j',k':
    sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k'*n+i',c] = 1 if (i==i' and j==j' and k==k') else 0
    """
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1 if (i==ip and j==jp and k==kp) else 0
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[jp*p+kp, c] * W[kp*n+ip, c]
                            if s != expected:
                                return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """
    Faster verification using numpy. Convert to float and check with tolerance,
    then do exact check if float passes.
    """
    r = len(U[0]) if U else 0
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k; ip,jp,kp] = sum_c U[i*m+j,c]*V[jp*p+kp,c]*W[kp*n+ip,c]
    # We want T == I_n ⊗ I_m ⊗ I_p (in appropriate flattening)
    
    # For efficiency, check each (i,j,k) target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # For target (i,j,k), check all (ip,jp,kp)
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            # Compute sum_c
                            u_val = Uf[i*m+j, :]  # shape (r,)
                            v_val = Vf[jp*p+kp, :]  # shape (r,)
                            w_val = Wf[kp*n+ip, :]  # shape (r,)
                            s = float(np.dot(u_val, v_val * w_val))
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, V, W  # Wait, that's wrong, should return U, V, W

def _strassen_2x2x2_fixed():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4 = a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline: rank 75
    rank_school = n * m * p  # 75
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Strassen 2x2x2 blocks
    # We can fit 4 disjoint 2x2x2 blocks in the 3x5x5 tensor
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Blocks:
    # 1. i:0-1, j:0-1, k:0-1
    # 2. i:0-1, j:2-3, k:0-1
    # 3. i:0-1, j:0-1, k:2-3
    # 4. i:0-1, j:2-3, k:2-3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # For each block, we replace 8 schoolbook terms with 7 Strassen terms.
    # We need to build new U, V, W with rank 75 - 4 = 71.
    
    new_rank = 71
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    # Identify which (i,j,k) are in blocks.
    in_block = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    in_block.add((i0+di, j0+dj, k0+dk))
    
    col_idx = 0
    # 1. Add schoolbook terms for non-block indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in in_block:
                    new_U[i*m+j][col_idx] = 1
                    new_V[j*p+k][col_idx] = 1
                    new_W[k*n+i][col_idx] = 1
                    col_idx += 1
    
    # 2. Add Strassen terms for each block
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2_fixed()
    
    for (i0, j0, k0) in blocks:
        for c in range(7):
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    u_val = strassen_U[li*2+lj][c]
                    if u_val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        new_U[gi*m+gj][col_idx] += u_val
            
            for lj in range(2):
                for lk in range(2):
                    v_val = strassen_V[lj*2+lk][c]
                    if v_val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        new_V[gj*p+gk][col_idx] += v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = strassen_W[lk*2+li][c]
                    if w_val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        new_W[gk*n+gi][col_idx] += w_val
            
            col_idx += 1
    
    # Verify using fast method
    if _verify_numpy(new_U, new_V, new_W, n, m, p):
        # Do exact verification
        if _verify_fast(new_U, new_V, new_W, n, m, p):
            best = {"U": new_U, "V": new_V, "W": new_W}
            best_rank = new_rank
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return best
    
    # Try to find even better decomposition
    # The remaining part after 4 Strassen blocks is:
    # - i in {0,1}, j=4, k in {0..4}: 2*1*5 = 10 terms (schoolbook)
    # - i in {0,1}, j in {0..3}, k=4: 2*4*1 = 8 terms (schoolbook)
    # - i=2, j in {0..4}, k in {0..4}: 1*5*5 = 25 terms (schoolbook)
    # Total: 43 terms, giving rank 71.
    
    # Can we do better on the i=2 row?
    # For i=2, we need to compute (A*B)_{2,k} for all k.
    # This is a linear combination of columns of B, weighted by row 2 of A.
    # The rank needed is at most 5*5=25 for this part alone.
    
    # What if we use a different approach for the whole thing?
    # Let's try to see if we can reduce the rank below 71.
    
    # One idea: use the fact that the tensor has some symmetries or structure.
    # For 3x5x5, the known lower bounds are around 55-65.
    
    # Let's try a random search for rank 70.
    # This is computationally expensive, so we'll try a limited search.
    
    if time.time() - start_time < 135:
        # Try rank 70
        try:
            r_target = 70
            # We need to find U, V, W of rank r_target
            # This is a hard problem. Let's use a simpler approach.
            
            # Actually, let's just return the 71-rank solution as it's proven to work.
            pass
        except Exception as e:
            pass
    
    return best

# EVOLVE-BLOCK-END

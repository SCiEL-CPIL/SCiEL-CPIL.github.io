"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 115.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _wftt_block(n, m, q):
    """
    WFTT for n x m * m x q using cyclic convolution over GF(5).
    Returns (U, V, W) lists of dicts for one block of q columns.
    n=3, m=3, q=4.
    """
    # WFTT: use polynomial multiplication mod x^q - 1
    # Represent A as vector a of length m, B as vector b of length q
    # Product c[i] = sum_j A[i,j] * B[j, :]
    
    # For the k-th output row i, c[i,:] = A[i,:] * B[:, :]
    # This is a convolution: c[i,k] = sum_j A[i,j] * B[j,k]
    
    # WFTT approach: evaluate at q points in GF(5)
    # A[i, x] = sum_j A[i,j] * x^j  (polynomial in x of degree < m)
    # B[x, k] = sum_k' B[k', k] * x^{-k'}  ... actually let's think differently
    
    # Standard WFTT for 1D convolution of length m and q:
    # We want c[k] = sum_{j=0}^{m-1} a[j] * b[k-j mod q] for k=0..q-1
    # But here b has length q and a has length m, and we want:
    # c[k] = sum_{j=0}^{m-1} a[j] * b[j, k]
    
    # Actually the tensor is: T[i,j,k] = A[i,j]*B[j,k]
    # For fixed i: c_i[k] = sum_j A[i,j] * B[j,k]
    
    # WFTT: Let omega be a primitive q-th root of unity in GF(5).
    # For q=4, we need 4th roots of unity in GF(5). 
    # GF(5): 1^4=1, 2^4=16=1, 3^4=81=1, 4^4=256=1. 
    # Actually 2^2=4=-1, so 2 is a 4th root of unity.
    # omega = 2, omega^2 = 4, omega^3 = 3, omega^4 = 1
    
    # WFTT steps:
    # 1. For each i, compute A_hat[i, t] = sum_j A[i,j] * omega^{j*t} for t=0..q-1
    # 2. For each k, compute B_hat[t, k] = sum_j B[j,k] * omega^{-j*t} for t=0..q-1
    # 3. Product: P[i, t, k] = A_hat[i,t] * B_hat[t,k]
    # 4. Inverse: c[i,k] = (1/q) * sum_t P[i,t,k] * omega^{t*k}
    
    # But we need this to work as a tensor decomposition.
    # The rank-1 terms correspond to each (i, t, k) triple... no.
    
    # Let me think about this more carefully for the tensor decomposition.
    
    # The matrix multiplication tensor for 3x3 * 3x4 is:
    # T[i,j,k] = sum over rank-1 terms: U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # WFTT construction:
    # For each t in {0,1,2,3} (frequency index):
    #   U_t[i,j] = A[i,j] * omega^{j*t}  (but this depends on A, which is input)
    
    # Wait, I'm confusing the algorithm with the tensor decomposition.
    # The tensor decomposition must be FIXED, not depending on the input matrices.
    
    # Let me reconsider. The WFTT gives an algorithm, but we need to extract
    # the tensor decomposition from it.
    
    # Actually, for the tensor decomposition, we can use the following:
    # The multiplication C = A*B where A is 3x3 and B is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using WFTT with omega=2 in GF(5):
    # Define for t=0,1,2,3:
    #   U_t: 3x3 matrix where U_t[i,j] = omega^{j*t}
    #   V_t: 3x4 matrix where V_t[j,k] = omega^{-j*t}  ... no, this doesn't work directly
    
    # Let me use a different formulation. The key insight is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # If we define:
    #   For each t: 
    #     A_t[i,j] = A[i,j] * omega^{j*t}
    #     B_t[j,k] = B[j,k] * omega^{-j*t}
    #     Then C[i,k] * omega^{?} ... this doesn't separate cleanly.
    
    # Let me try the standard approach:
    # C[i,k] = (1/q) sum_t [sum_j A[i,j] omega^{j*t}] * [sum_j B[j,k] omega^{-j*t}] * omega^{t*k}
    # 
    # Wait, that's not right either. Let me be more careful.
    
    # Actually, the correct WFTT decomposition for the tensor is:
    # T[i,j,k] (the structure tensor) can be decomposed as:
    # For each t in {0,...,q-1}:
    #   The t-th "slice" contributes rank-1 terms
    
    # Let me just construct it directly using the fact that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using the identity: sum_j x_j y_j = (1/q) sum_t (sum_j x_j omega^{jt})(sum_j y_j omega^{-jt})
    # where the sums are over j=0..q-1 (padding if needed).
    
    # But here j only goes 0..m-1=2, and we're summing over j, not k.
    # And k is independent. So:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # This is just a dot product over j for each (i,k). The WFTT applies to
    # convolutions, not simple dot products. For a simple dot product of length m,
    # the WFTT doesn't reduce the rank below m (it's already m).
    
    # So WFTT doesn't help here for the m=3 dimension. Let me abandon this approach
    # and try something else.
    
    # Alternative: Use the fact that 3x3 * 3x8 can be decomposed as:
    # - 3x3 * 3x4 (first 4 columns)
    # - 3x3 * 3x4 (last 4 columns)
    # Each 3x3 * 3x4 has rank at most 3*4=12 schoolbook, but maybe less.
    
    # For 3x3 * 3x4, the tensor has 3*3*4 = 36 entries.
    # The minimum rank is at least ceil(36/something)... 
    # Actually, for n x m * m x p, the rank is at least max(n*p, m*max(n,p), ...).
    # For 3x3*3x4: rank >= 3*4=12? No, that's not a lower bound.
    # The trivial lower bound is max(n, m, p) = 3, but that's weak.
    # Actually rank >= n*p/m if m divides... no.
    
    # Let me just try to find a good decomposition for 3x3*3x4 and duplicate it.
    
    # For now, return schoolbook for this block
    r = n * m * q
    U = [{i*m+j: 1} for i in range(n) for j in range(m) for k in range(q)]
    V = [{j*q+k: 1} for i in range(n) for j in range(m) for k in range(q)]
    W = [{k*n+i: 1} for i in range(n) for j in range(m) for k in range(q)]
    
    # Actually let me just return None and use schoolbook
    return None

def _construct_3x3x4():
    """
    Try to find a rank-minimal decomposition for 3x3 * 3x4.
    Schoolbook rank is 36. Let's see if we can do better.
    
    The tensor T[i,j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    has 36 entries, all equal to 1 in the structure tensor.
    
    A rank-1 term is specified by U (9 entries), V (12 entries), W (12 entries)
    such that for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    
    Let's try a Strassen-like approach:
    Split the 4 columns into two groups of 2.
    For each group, use a 3x3 * 3x2 decomposition.
    
    3x3 * 3x2: schoolbook rank 18.
    Can we do better? 
    
    For 3x3 * 3x2:
    C[i,k] = sum_j A[i,j] B[j,k] for k=0,1
    
    This is equivalent to computing two dot products of length 3 for each i.
    Total: 3*2 = 6 dot products of length 3.
    Schoolbook: 18 multiplications.
    
    Using Strassen-like: for each i, compute:
    s0 = A[i,0] + A[i,1]
    s1 = A[i,1] + A[i,2]  
    s2 = A[i,0] + A[i,2]
    
    Then:
    A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    = ... this doesn't directly help.
    
    Actually, for a single row of A (length 3) times B (3x2), we need:
    c0 = a0*b00 + a1*b10 + a2*b20
    c1 = a0*b01 + a1*b11 + a2*b21
    
    This is 6 scalar multiplications. Can we do it in fewer?
    The rank of the 3x2 matrix multiplication tensor (for one row) is 3 (since it's just a vector-matrix product, rank = min(3,2) = 2? No, the tensor rank of a 1x3 * 3x2 product is 3, since you need 3 multiplications).
    
    Actually no. The product of a 1x3 vector and a 3x2 matrix gives a 1x2 result.
    The tensor is T[a0,a1,a2, b00,b01,b10,b11,b20,b21] -> (c0, c1)
    where c0 = a0*b00 + a1*b10 + a2*b20
    c1 = a0*b01 + a1*b11 + a2*b21
    
    This can be done with 6 multiplications (schoolbook). Can we do it in 5?
    The rank of this tensor... I believe it's 6 for generic matrices, but let's not go down this path.
    
    Let me try a different approach: use the full 3x3 * 3x8 tensor and look for
    patterns that share computations across the 8 columns.
    """
    n, m, p = 3, 3, 8
    
    # Let's try a direct construction using the following idea:
    # For each pair of rows (i1, i2) in A, and each column k in B,
    # we can compute certain combinations.
    
    # Actually, let's just use schoolbook but try to find a better decomposition
    # using a simple greedy approach with a limited candidate set.
    
    # Generate candidates: each rank-1 term has U, V, W with small support.
    # U support: subset of {0..8} (9 positions)
    # V support: subset of {0..23} (24 positions)  
    # W support: subset of {0..23} (24 positions)
    
    # For efficiency, only consider terms where:
    # - U has support of size 1 or 2
    # - V has support of size 1 or 2
    # - W has support of size 1 or 2
    # - Coefficients are in {-1, 1}
    
    # This gives a manageable number of candidates.
    
    candidates = []
    
    # Add schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i*m+j: 1}
                v = {j*p+k: 1}
                w = {k*n+i: 1}
                candidates.append((u, v, w))
    
    # Add terms with U having 2 elements
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1,j1) >= (i2,j2):
                        continue
                    for sign1 in [1]:
                        for sign2 in [1, -1]:
                            u = {i1*m+j1: sign1, i2*m+j2: sign2}
                            for j in range(m):
                                for k in range(p):
                                    for i in range(n):
                                        v = {j*p+k: 1}
                                        w = {k*n+i: 1}
                                        candidates.append((u, v, w))
    
    # Add terms with V having 2 elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    for k2 in range(p):
                        if (j,k) >= (j2,k2):
                            continue
                        for sign1 in [1]:
                            for sign2 in [1, -1]:
                                u = {i*m+j: 1}
                                v = {j*p+k: sign1, j2*p+k2: sign2}
                                w = {k*n+i: 1}
                                candidates.append((u, v, w))
    
    # Add terms with W having 2 elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i2 in range(n):
                        if (k,i) >= (k2,i2):
                            continue
                        for sign1 in [1]:
                            for sign2 in [1, -1]:
                                u = {i*m+j: 1}
                                v = {j*p+k: 1}
                                w = {k*n+i: sign1, k2*n+i2: sign2}
                                candidates.append((u, v, w))
    
    # Add terms with U and V each having 2 elements
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1,j1) >= (i2,j2):
                        continue
                    for j in range(m):
                        for k in range(p):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if (j,k) >= (j2,k2):
                                        continue
                                    for i in range(n):
                                        u = {i1*m+j1: 1, i2*m+j2: 1}
                                        v = {j*p+k: 1, j2*p+k2: 1}
                                        w = {k*n+i: 1}
                                        candidates.append((u, v, w))
    
    # Now do greedy set cover
    current = np.zeros((n, m, p))
    selected = []
    remaining = set([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
    
    max_iter = 36  # schoolbook rank
    for iteration in range(max_iter):
        if not remaining or _time_left() < 5.0:
            break
        
        best_idx = -1
        best_score = -1
        best_contrib = None
        
        for c_idx, (u, v, w) in enumerate(candidates):
            score = 0
            contrib = {}
            for (i, j, k) in remaining:
                val = u.get(i*m+j, 0) * v.get(j*p+k, 0) * w.get(k*n+i, 0)
                if val != 0:
                    cur = current[i, j, k]
                    if abs(cur + val - 1) < abs(cur - 1) - 1e-10:
                        score += 1
                        contrib[(i,j,k)] = val
            
            if score > best_score:
                best_score = score
                best_idx = c_idx
                best_contrib = contrib
        
        if best_idx < 0 or best_score <= 0:
            break
        
        u, v, w = candidates[best_idx]
        selected.append(best_idx)
        
        new_remaining = set()
        for (i, j, k) in remaining:
            val = best_contrib.get((i,j,k), 0)
            current[i, j, k] += val
            if abs(current[i, j, k] - 1) > 1e-8:
                new_remaining.add((i, j, k))
            else:
                current[i, j, k] = 1.0
        
        remaining = new_remaining
    
    if remaining:
        return None
    
    r = len(selected)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c_idx, cand_idx in enumerate(selected):
        u, v, w = candidates[cand_idx]
        for idx, val in u.items():
            U[idx][c_idx] += val
        for idx, val in v.items():
            V[idx][c_idx] += val
        for idx, val in w.items():
            W[idx][c_idx] += val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try the direct construction
    if _time_left() > 10.0:
        result = _construct_3x3x4()
        if result is not None and _verify(result["U"], result["V"], result["W"], n, m, p):
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END

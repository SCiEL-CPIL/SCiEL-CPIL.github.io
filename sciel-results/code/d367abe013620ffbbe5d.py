"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_3x3():
    """
    Return Strassen's rank-7 decomposition for 3x3 matrix multiplication.
    For C = A * B where A, B are 3x3:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    The 7 products are:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    Then:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For 3x3, indices: row i in {0,1,2}, col j in {0,1,2}, k in {0,1,2}
    # We need to handle all 9 entries, not just 4.
    # Let's use a different approach: embed 3x3 into 4x4 and use Strassen,
    # or find a direct rank-7 (or lower) decomposition for 3x3.
    
    # Actually, for 3x3 matrix multiplication, the tensor rank is known to be 19 (not 7).
    # Strassen's algorithm works for 2x2 with rank 7. For 3x3, we need a different approach.
    
    # Let's instead use the fact that we can compute A * B for 3x3 * 3x3 
    # by splitting into 2x2 blocks and using Strassen recursively, but that's complex.
    
    # Alternative: Use the known minimal rank for 3x3 matrix mult.
    # The tensor rank of 3x3 matrix multiplication is 19 (Landsberg 1988).
    # But we need integer/half-integer entries.
    
    # For now, let's return None and use a different strategy.
    return None


def _build_strassen_init(n=3, m=3, p=6):
    """
    Build an initial decomposition using block structure.
    Split B (3x6) into three 3x3 blocks: B[:,0:3], B[:,3:6]
    C = A * B = [A*B1, A*B2] where B1, B2 are 3x3.
    
    We can use separate Strassen-like decompositions for each block,
    but share the U factors where possible.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # For 3x3 * 3x3, use a known low-rank decomposition.
    # The rank of 3x3 matrix multiplication tensor is 19.
    # Let's use a simple structured initialization.
    
    # Strategy: Use 2 separate rank-19 decompositions for the two 3x3 blocks,
    # giving total rank 38. Then try to merge/optimize.
    
    # Actually, let's try a different approach:
    # View the problem as computing C[i,k] = sum_j A[i,j] * B[j,k] for k=0..5
    # Group k into two sets: k=0,1,2 and k=3,4,5
    
    # For each group, we have a 3x3 * 3x3 multiplication.
    # Use the same U pattern for both, but different V and W.
    
    # Simple structured init: for each (i,j) pair in A, and each k in B,
    # create a rank-1 term. That's schoolbook rank 54.
    
    # Better: use the structure that A[i,j] multiplies the entire row j of B.
    # For fixed i, C[i,:] = sum_j A[i,j] * B[j,:]
    # This is a linear combination of rows of B.
    
    # Let's try rank 36: for each i (3 values) and each j (3 values), 
    # we need to compute A[i,j] * B[j,0:6] and add to C[i,0:6].
    # That's 9 terms, each of rank 6 (since B[j,:] has 6 elements).
    # Total rank = 9 * 6 = 54 (schoolbook).
    
    # Alternative: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # We can precompute linear combinations of rows of B.
    # Let S[i,:] = sum_j A[i,j] * B[j,:]
    # This is just matrix-vector multiplication for each row.
    
    # The key insight: the tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # For fixed i, the terms are: for j=0,1,2 and k=0..5:
    #   a = i*3+j, b = j*6+k, c = k*3+i
    # So for fixed i, we have 3*6 = 18 nonzero entries.
    # For all i, total 3*18 = 54 nonzero entries.
    
    # The tensor has a block structure: for each i, the sub-tensor over (j,k) is
    # a 3x6 matrix that's all ones (in the appropriate positions).
    
    # Let's try to find a rank-36 decomposition by exploiting the i-block structure.
    # For each i, we need to represent the 3x6 "all ones" pattern in the (j,k) space
    # as a sum of rank-1 terms in the (j,k) -> (j*6+k, k*3+i) mapping.
    
    # Actually, let's just use ALS with a good initialization.
    
    # Initialize with schoolbook, then try to reduce rank.
    return None


def _try_als_decomposition(n=3, m=3, p=6, deadline=None):
    """
    Try to find a low-rank decomposition using ALS with various initializations.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = None
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But the actual tensor rank is likely higher.
    # For 3x3 * 3x6, the rank is at least 18 (from dimension constraints).
    # Let's try ranks from 18 upward.
    
    for r in range(18, 40):
        if time.time() > deadline - 10:
            break
        
        n_restarts = max(2, int((deadline - time.time()) / 15))
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initializations
            if restart == 0:
                # Sparse random initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for col in range(r):
                    # Assign random nonzero entries
                    n_u = np.random.randint(1, 4)
                    n_v = np.random.randint(1, 4)
                    n_w = np.random.randint(1, 4)
                    u_idx = np.random.choice(nm, n_u, replace=False)
                    v_idx = np.random.choice(mp, n_v, replace=False)
                    w_idx = np.random.choice(pn, n_w, replace=False)
                    U[u_idx, col] = np.random.choice([-1, 1], n_u)
                    V[v_idx, col] = np.random.choice([-1, 1], n_v)
                    W[w_idx, col] = np.random.choice([-1, 1], n_w)
            elif restart == 1:
                # Dense random initialization
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
            else:
                # Mixed initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(pn, r) * 0.3
                # Add some structure: make U depend on i, V on j, W on k
                for a in range(nm):
                    i_a = a // m
                    U[a, :] *= (1.0 + 0.1 * i_a)
                for b in range(mp):
                    j_b = b // p
                    V[b, :] *= (1.0 + 0.1 * j_b)
                for c in range(pn):
                    k_c = c // n
                    W[c, :] *= (1.0 + 0.1 * k_c)
            
            # ALS iterations
            max_iter = 80
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                new_U = np.zeros_like(U)
                for a in range(nm):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == a:
                                    b_idx = j * p + k
                                    c_idx = k * n + i
                                    rows.append(V[b_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_U[a, :] = U[a, :]
                U = new_U
                
                # Update V
                new_V = np.zeros_like(V)
                for b in range(mp):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if j * p + k == b:
                                    a_idx = i * m + j
                                    c_idx = k * n + i
                                    rows.append(U[a_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_V[b, :] = V[b, :]
                V = new_V
                
                # Update W
                new_W = np.zeros_like(W)
                for c in range(pn):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if k * n + i == c:
                                    a_idx = i * m + j
                                    b_idx = j * p + k
                                    rows.append(U[a_idx, :] * V[b_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_W[c, :] = W[c, :]
                W = new_W
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                # Check off-support (sampled for speed)
                if it % 10 == 9:
                    for _ in range(100):
                        a = np.random.randint(0, nm)
                        b = np.random.randint(0, mp)
                        c = np.random.randint(0, pn)
                        i_a, j_a = a // m, a % m
                        j_b, k_b = b // p, b % p
                        k_c, i_c = c // n, c % n
                        if not (j_a == j_b and k_b == k_c and i_c == i_a):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += val ** 2
                
                if err < 1e-6:
                    # Verify exactly
                    exact_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                exact_err += (val - 1.0) ** 2
                    for a in range(nm):
                        i_a, j_a = a // m, a % m
                        for b in range(mp):
                            j_b, k_b = b // p, b % p
                            for c in range(pn):
                                k_c, i_c = c // n, c % n
                                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                                    exact_err += val ** 2
                    
                    if exact_err < 1e-6:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
        
        if best_result is not None:
            break
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to find better
        better = _try_als_decomposition(n, m, p, deadline)
        if better is not None:
            rank_better = len(better["U"][0]) if better["U"] else 0
            rank_current = len(result["U"][0]) if result["U"] else 0
            if rank_better < rank_current:
                result = better
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END

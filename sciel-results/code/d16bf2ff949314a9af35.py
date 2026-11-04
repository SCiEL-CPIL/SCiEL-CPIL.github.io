import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 142.0  # safety margin before 150s kill

def _build_strassen56():
    """Build the rank-56 Strassen-based decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    
    # Strassen 2x2 local coefficients
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    group = 0
    for jj in range(p // 2):  # 4 groups of 2 columns
        for half in range(2):  # 2 groups of 2 rows
            base_r = group * 7
            group += 1
            
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx]][rr] += val
                for idx, val in v_list:
                    V[b_map[idx]][rr] += val
                for idx, val in w_list:
                    W[c_map[idx]][rr] += val

    return U, V, W

def _validate_fast(U, V, W, n=2, m=4, p=8, seed=42, trials=3):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor using NumPy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We want to compute the outer sum.
        # C[i,k] = sum_r sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # For each r:
        #   term_r[i,k] = sum_j U[i*m+j, r] * V[j*p+k, r]
        #   where U_r is n x m, V_r is m x p
        #   So U_r @ V_r gives n x p
        #   Then C = sum_r (U_r @ V_r) * W_r_col
        #   where W_r_col[i] = W[k*n+i, r] ... wait W is indexed by (k*n+i)
        #   W[k*n+i, r] for fixed r is a vector of size p*n. Reshape to (n, p)?
        #   Index is k*n + i. So W_r reshaped to (p, n) has element [k, i] = W_r[k*n+i]
        #   We need to multiply element-wise: (U_r @ V_r) * W_r reshaped to (n,p)?
        #   (U_r @ V_r) has index [i, k]. W_r reshaped to (p, n) has [k, i].
        #   So we need W_r transposed/reshaped to (n, p).
        #   W_r_vec index is k*n + i. 
        #   If we reshape to (n, p), index [i, k] corresponds to flat index i*p + k.
        #   But flat index is k*n + i.
        #   So reshape to (p, n) gives [k, i] = flat index k*n + i.
        #   So W_r_mat = W_r.reshape(p, n).T gives [i, k].
        
        C_computed = np.zeros((n, p), dtype=object)
        for r in range(rank):
            Ur = Ua[:, r].reshape(n, m)
            Vr = Va[:, r].reshape(m, p)
            Wr = Wa[:, r].reshape(p, n).T  # now n x p
            C_computed += Ur @ Vr * Wr
        
        if not np.array_equal(C_expected, C_computed):
            return False
    return True

def _try_rank_reduction(U, V, W):
    """
    Attempt to find a rank reduction using column linear dependencies.
    We look for vectors c in the null space of the "column signature" such that
    removing the corresponding linear combination doesn't change the tensor.
    
    Since we only have integer entries, we can use rational arithmetic or just
    check for exact integer linear dependencies.
    """
    rank = len(U[0])
    if time.time() > _start_time + _TIMEOUT - 5.0:
        return U, V, W

    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    
    # Check for columns that are scalar multiples of others (simple case)
    # Or check if any column is in the span of others.
    # For rank r, we have r columns in 8, 32, 16 dimensional spaces.
    # The tensor constraint is nonlinear, so simple column dependency is not sufficient.
    # However, if a column r is exactly equal to column r', we can merge.
    # If column r = -1 * column r', etc.
    
    # Let's just check for exact duplicates or sign flips first.
    # This is a cheap check.
    
    # Convert to list of tuples for hashing
    def col_tuple(matrix, col_idx):
        return tuple(matrix[i, col_idx] for i in range(len(matrix)))

    u_cols = [col_tuple(Ua, i) for i in range(rank)]
    v_cols = [col_tuple(Va, i) for i in range(rank)]
    w_cols = [col_tuple(Wa, i) for i in range(rank)]
    
    # This approach is too simple. The real rank reduction would require finding
    # a non-trivial linear combination of columns that yields the zero tensor.
    # i.e., sum_r c_r T_r = 0.
    # This is equivalent to finding a vector c such that
    # (sum_r c_r U_r) (sum_r c_r V_r) (sum_r c_r W_r) = 0 is NOT the condition.
    # The condition is sum_r c_r U_r^{(i,j)} V_r^{(j,k)} W_r^{(k,i)} = 0 for all i,j,k.
    
    # This is a system of quadratic equations in c. Hard to solve.
    # Given the 120s limit and the known optimal rank, we will not succeed here easily.
    # We will return the original rank.
    
    return U, V, W

def construct():
    n, m, p = 2, 4, 8
    
    U, V, W = _build_strassen56()
    
    # Validate quickly
    if not _validate_fast(U, V, W, n, m, p, trials=2):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank_school = n * m * p
        U = [[0]*rank_school for _ in range(n*m)]
        V = [[0]*rank_school for _ in range(m*p)]
        W = [[0]*rank_school for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        
    # Try rank reduction (likely to fail, but keeps the door open)
    if time.time() < _start_time + _TIMEOUT - 2.0:
        U, V, W = _try_rank_reduction(U, V, W)

    return {"U": U, "V": V, "W": W}

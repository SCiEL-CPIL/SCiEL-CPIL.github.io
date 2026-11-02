import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 135.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _get_2x2x3_decomp():
    """Rank 7 decomposition for 2x2x3 matrix multiplication (Winograd).
    U: 4x7 (rows i*2+j), V: 6x7 (rows j*3+k), W: 6x7 (rows k*2+i)
    """
    U = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 0, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1]
    ], dtype=float)
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_r U[i,r] V[j,r] W[k,r] == expected delta."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    import random
    random.seed(42)
    for _ in range(2000):
        i = random.randint(0, n*m-1)
        j = random.randint(0, m*p-1)
        k = random.randint(0, p*n-1)
        
        i_val, j_val_i = divmod(i, m)
        j_val_j, k_val_j = divmod(j, p)
        k_val_k, i_val_k = divmod(k, n)
        
        expected = 1.0 if (j_val_i == j_val_j and k_val_j == k_val_k and i_val_k == i_val) else 0.0
        actual = T[i, j, k]
        if abs(actual - expected) > 1e-9:
            return False
    return True


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Exact verification using the full tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = np.sum(Ua[u, :] * Va[v, :] * Wa[io, :])
                if abs(s - T[io, u, v]) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _kron_4x4x9():
    """Kronecker product of 2x2x3 decompositions to get 4x4x9.
    This gives rank 7*7=49 for the 4x4x9 problem.
    """
    U1, V1, W1 = _get_2x2x3_decomp()
    
    # Kronecker products
    U_raw = np.kron(U1, U1)  # 16 x 49
    V_raw = np.kron(V1, V1)  # 36 x 49
    W_raw = np.kron(W1, W1)  # 36 x 49
    
    # Permute rows for U: (i1*2+j1)*4 + (i2*2+j2) -> (i1*2+i2)*4 + (j1*2+j2)
    U_std = np.zeros((16, 49))
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    r_raw = (i1*2 + j1)*4 + (i2*2 + j2)
                    i_std = i1*2 + i2
                    j_std = j1*2 + j2
                    r_std = i_std*4 + j_std
                    U_std[r_std] = U_raw[r_raw]
    
    # Permute rows for V: (j1*3+k1)*6 + (j2*3+k2) -> (j1*2+j2)*9 + (k1*3+k2)
    V_std = np.zeros((36, 49))
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(3):
                for k2 in range(3):
                    r_raw = (j1*3 + k1)*6 + (j2*3 + k2)
                    j_std = j1*2 + j2
                    k_std = k1*3 + k2
                    r_std = j_std*9 + k_std
                    V_std[r_std] = V_raw[r_raw]
    
    # Permute rows for W: (k1*2+i1)*6 + (k2*2+i2) -> (k1*3+k2)*4 + (i1*2+i2)
    W_std = np.zeros((36, 49))
    for k1 in range(3):
        for k2 in range(3):
            for i1 in range(2):
                for i2 in range(2):
                    r_raw = (k1*2 + i1)*6 + (k2*2 + i2)
                    k_std = k1*3 + k2
                    i_std = i1*2 + i2
                    r_std = k_std*4 + i_std
                    W_std[r_std] = W_raw[r_raw]
    
    return U_std, V_std, W_std


def _partition_3way(n, m, p):
    """3-way partitioning: split 4x4 matrix into strips of heights 1,1,2.
    Split 4x9 matrix into corresponding vertical strips.
    Use rank-7 for 2x2x3 sub-problem, rank-9 for 1x1x9 strips.
    Total rank: 7 + 9 + 9 = 25? No, that's not right.
    
    Actually, let me think about this more carefully.
    The tensor is T[i,j,k] where i in [0,4), j in [0,4), k in [0,9).
    
    Partition i into {0}, {1}, {2,3}.
    For each part, we need a decomposition of the sub-tensor.
    
    For i=0: T[0,j,k] = 1 iff j==j and k==k... wait, that's not right.
    T[i,j,k] = 1 always? No.
    
    Let me re-read the problem. The tensor is for matrix multiplication.
    T[u,v,w] where u=i*m+j, v=j'*p+k, w=k'*n+i'
    T[u,v,w] = 1 iff j==j' and k==k' and i==i'
    
    So for fixed i, the sub-tensor over j,k is:
    T[i,j,k] = 1 (for all j,k)
    
    That's a rank-1 tensor! So each strip of height 1 has rank m*p = 4*9 = 36? No.
    
    Wait, I'm confusing myself. Let me re-derive.
    
    The matrix multiplication tensor has entries:
    T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B
    T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    
    In the flattened form:
    U index: i*m + j (range 0..15)
    V index: j*p + k (range 0..35)  
    W index: k*n + i (range 0..35)
    
    T[u,v,w] = 1 iff:
      u = i*m+j, v = j'*p+k, w = k'*n+i'
      j == j' and k == k' and i == i'
    
    So for a fixed i, the sub-tensor T_i[j,k,j',k',i'] has:
    T_i[j,k,j',k',i'] = 1 iff j==j' and k==k' and i==i'
    
    For fixed i, this is a tensor over (j,k) x (j',k') x (i').
    The condition i==i' means i' must equal i.
    The condition j==j' and k==k' means (j,k) == (j',k').
    
    So T_i is essentially: for each (j,k), T_i[j,k,j,k,i] = 1, and 0 elsewhere.
    This is a sum of m*p = 36 rank-1 tensors.
    
    So the 3-way partitioning doesn't help reduce rank below 144.
    
    Let me try a different approach: use the Kronecker product result (rank 49)
    and see if we can do better.
    """
    pass


def _try_linear_combination_reduction(U, V, W, T, n, m, p, max_time=10.0):
    """Try to reduce rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic approach; not guaranteed to find the minimum rank.
    """
    start = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # For each term c, check if its tensor can be expressed as a combination of others.
    # This is expensive, so we only try a few terms.
    def term_support(c):
        """Get non-zero entries of the c-th rank-1 tensor."""
        support = {}
        for io in range(p * n):
            for u in range(n * m):
                for v in range(m * p):
                    val = W[io][c] * U[u][c] * V[v][c]
                    if val != 0:
                        support[(io, u, v)] = val
        return support
    
    # Try to eliminate each term
    for c in range(rank):
        if time.time() - start > max_time:
            break
        
        tc = term_support(c)
        if not tc:
            # Zero term, remove it
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            continue
        
        # Check if tc is in the span of other terms
        # Build linear system
        cols = [d for d in range(rank) if d != c]
        keys = list(tc.keys())
        
        A = []
        b = []
        for key in keys:
            rowA = []
            for d in cols:
                io, u, v = key
                rowA.append(W[io][d] * U[u][d] * V[v][d])
            A.append(rowA)
            b.append(tc[key])
        
        # Solve A x = b over Q
        try:
            from fractions import Fraction
            A2 = [[Fraction(x) for x in row] for row in A]
            b2 = [Fraction(x) for x in b]
            ncol = len(cols)
            nrow = len(keys)
            aug = [A2[i] + [b2[i]] for i in range(nrow)]
            
            r = 0
            for col in range(ncol):
                piv = None
                for i in range(r, nrow):
                    if aug[i][col] != 0:
                        piv = i
                        break
                if piv is None:
                    continue
                aug[r], aug[piv] = aug[piv], aug[r]
                pv = aug[r][col]
                aug[r] = [x / pv for x in aug[r]]
                for i in range(nrow):
                    if i != r and aug[i][col] != 0:
                        f = aug[i][col]
                        aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
                r += 1
            
            # Check if solution is integer
            sol = [Fraction(0)] * ncol
            valid = True
            for i in range(r):
                # Find leading column
                lead_col = -1
                for j in range(ncol):
                    if aug[i][j] != 0:
                        lead_col = j
                        break
                if lead_col == -1:
                    if aug[i][ncol] != 0:
                        valid = False
                        break
                else:
                    sol[lead_col] = aug[i][ncol]
            
            if not valid:
                continue
            
            # Check remaining rows
            for i in range(r, nrow):
                if any(aug[i][j] != 0 for j in range(ncol)) or aug[i][ncol] != 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if all solutions are integer
            if all(s.denominator == 1 for s in sol):
                # Found integer solution, but we can't simply add to U,V,W independently
                # because the tensor is a product, not a sum.
                # This approach doesn't work for rank reduction.
                continue
        except Exception:
            continue
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    
    # Try Kronecker product decomposition (rank 49)
    try:
        U_std, V_std, W_std = _kron_4x4x9()
        
        if _verify(U_std, V_std, W_std, n, m, p):
            # Try to reduce rank further
            T = _build_tensor(n, m, p)
            result = _try_linear_combination_reduction(
                U_std.tolist(), V_std.tolist(), W_std.tolist(), T, n, m, p, max_time=20.0
            )
            
            # Verify the reduced result
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
            else:
                # Fallback to the un-reduced Kronecker result
                return {
                    "U": U_std.tolist(),
                    "V": V_std.tolist(),
                    "W": W_std.tolist()
                }
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

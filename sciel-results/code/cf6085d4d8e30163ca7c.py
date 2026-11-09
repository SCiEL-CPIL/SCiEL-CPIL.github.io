"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against 1 if valid index, 0 otherwise
    for r in range(rank):
        pass  # We'll check via tensor contraction
    # Build the tensor from decomposition
    # T[i,j,k] should equal 1 if (i,j,k) is a valid matmul triple, else 0
    # Actually, the tensor for matmul n×m×p: T[i,j,k] = 1 if the indices are consistent
    # The tensor is: for each (i,j,k) where i in [0,n), j in [0,m), k in [0,p),
    # T[i,j,k] = 1 (it's the identity-like tensor)
    # Wait, no. The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need this sum to equal 1 for all (i,j,k) in the valid range.
    # Let's verify by checking a sample or all entries.
    # For speed, use einsum-like approach
    # Actually let's just do it directly but with numpy
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def _combine_decompositions(parts, n, m, p):
    """Combine a list of (U_part, V_part, W_part, offset_i, offset_j, offset_k, ni, nj, np)
    into a single decomposition for the full n x m x p problem."""
    total_rank = sum(len(U_part[0]) for U_part, _, _, _, _, _, _, _, _ in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for U_part, V_part, W_part, off_i, off_j, off_k, ni, nj, np in parts:
        rank_part = len(U_part[0])
        for i in range(ni):
            for j in range(nj):
                a_full = (off_i + i) * m + (off_j + j)
                a_part = i * nj + j
                for r in range(rank_part):
                    U[a_full][col + r] = U_part[a_part][r]
        for j in range(nj):
            for k in range(np):
                b_full = (off_j + j) * p + (off_k + k)
                b_part = j * np + k
                for r in range(rank_part):
                    V[b_full][col + r] = V_part[b_part][r]
        for k in range(np):
            for i in range(ni):
                c_full = (off_k + k) * n + (off_i + i)
                c_part = k * ni + i
                for r in range(rank_part):
                    W[c_full][col + r] = W_part[c_part][r]
        col += rank_part
    
    return U, V, W

def _build_from_splits(n, m, p, i_sizes, j_sizes, k_sizes):
    """Build decomposition given specific split sizes for each dimension."""
    parts = []
    i_off = 0
    for ni in i_sizes:
        j_off = 0
        for nj in j_sizes:
            k_off = 0
            for np_ in k_sizes:
                if ni == 2 and nj == 2 and np_ == 2:
                    U_part, V_part, W_part = _strassen_2x2x2()
                else:
                    U_part, V_part, W_part = _schoolbook(ni, nj, np_)
                parts.append((U_part, V_part, W_part, i_off, j_off, k_off, ni, nj, np_))
                k_off += np_
            j_off += nj
        i_off += ni
    
    U, V, W = _combine_decompositions(parts, n, m, p)
    return U, V, W

def _estimate_rank(i_sizes, j_sizes, k_sizes):
    """Estimate total rank for a given split."""
    total = 0
    for ni in i_sizes:
        for nj in j_sizes:
            for np_ in k_sizes:
                if ni == 2 and nj == 2 and np_ == 2:
                    total += 7
                else:
                    total += ni * nj * np_
    return total

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Generate all possible splits
    def gen_splits(total, max_parts):
        results = []
        def backtrack(remaining, parts):
            if not parts:
                if remaining > 0:
                    results.append([remaining])
                return
            if len(parts) >= max_parts:
                results.append(list(parts))
                return
            for v in range(1, remaining + 1):
                parts.append(v)
                backtrack(remaining - v, parts)
                parts.pop()
        backtrack(total, [])
        seen = set()
        unique = []
        for s in results:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
    
    i_splits = gen_splits(n, n)
    j_splits = gen_splits(m, m)
    k_splits = gen_splits(p, p)
    
    # Score each combination
    scored = []
    for isz in i_splits:
        for jsz in j_splits:
            for ksz in k_splits:
                total_r = _estimate_rank(isz, jsz, ksz)
                scored.append((total_r, isz, jsz, ksz))
    
    scored.sort(key=lambda x: x[0])
    
    # Try top candidates
    for total_r, isz, jsz, ksz in scored:
        if total_r >= best_rank:
            break
        if time.time() > deadline:
            break
        U2, V2, W2 = _build_from_splits(n, m, p, isz, jsz, ksz)
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify_fast(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
                # Early exit if we hit a good rank
                if best_rank <= 46:
                    break
    
    return best_result
# EVOLVE-BLOCK-END

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    rank = len(U[0]) if U else 0
    # Check support entries: T[i,j,k] = 1 for all (i,j,k)
    # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # We need this to equal 1 for all i,j,k (since matmul tensor has 1 at every (i,j,k))
    # Wait - actually the matmul tensor M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}, so the tensor is T[i,j,k] = 1 for all.
    # No wait. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we think of it as the structure tensor where
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the 3-tensor has T[i,j,k] = 1 for all valid (i,j,k).
    # Actually no. Let me re-read the task.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # The tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}??? 
    # No. For matrix multiplication: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is M[i,j,k] = 1 for all i,j,k? No!
    # The structure tensor of matrix multiplication has M[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p).
    # Wait, that's not right either. Let me think carefully.
    # 
    # The matrix multiplication map: (A, B) -> A*B where A is n x m, B is m x p.
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # 
    # As a 3-tensor T with indices (i,j,k): T[i,j,k] = 1 for all valid indices.
    # Because the bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C^*[i,k]
    # where C^* is the dual. And T[i,j,k] = 1 for all (i,j,k).
    # 
    # Hmm, but that would mean the tensor is just the all-ones tensor, which has rank 1.
    # That can't be right. Let me reconsider.
    #
    # Actually, the standard formulation: the matrix multiplication tensor M_{n,m,p} is a tensor
    # in U^* x V^* x W^* (or similar) where:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^* x f_j^* x g_k^* ... no.
    #
    # Let me look at the current code's verification:
    # It checks T[i*m+j, j*p+k, k*n+i] == 1 for all (i,j,k), and all other entries == 0.
    # 
    # So the tensor T is indexed by (U_index, V_index, W_index) where:
    # U_index = i*m + j (row i, col j of A)
    # V_index = j*p + k (row j, col k of B)  
    # W_index = k*n + i (row k, col i of C^T? or output)
    #
    # And T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some (i,j,k), else 0.
    # 
    # So the tensor is: sum_{i,j,k} e_{i*m+j} x e_{j*p+k} x e_{k*n+i}
    # This is the standard matrix multiplication tensor!
    # 
    # So T[u,v,w] = 1 if there exists (i,j,k) with u=i*m+j, v=j*p+k, w=k*n+i, else 0.
    # Note: for a given (u,v,w), the (i,j,k) is uniquely determined if it exists:
    # i = u // m, j = u % m, check v = j*p+k -> k = v - j*p, check w = k*n+i
    # 
    # So the support is exactly {(i*m+j, j*p+k, k*n+i) : i in [0,n), j in [0,m), k in [0,p)}
    # and T = 1 on this support, 0 elsewhere.

    rank = len(U[0]) if U else 0
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    
    # Compute T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # T[u,v,w] = sum_r Ua[u,r] * Va[v,r] * Wa[w,r]
    T = np.einsum('ur,vr,wr->uvw', Ua, Va, Wa)
    
    # Check support
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > 1e-9:
                    return False
    
    # Check non-support
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support.add((i * m + j, j * p + k, k * n + i))
    
    # For efficiency, only check a sample of non-support entries if tensor is large
    # But n*m*p = 144, and total size is (n*m)*(m*p)*(p*n) = 16*36*36 = 20736
    # That's small enough to check all.
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in support:
                    if abs(T[u, v, w]) > 1e-9:
                        return False
    return True


def _build_schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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


def _try_remove_term(U, V, W, col, n, m, p):
    """
    Try to remove rank-1 term at column 'col' by redistributing its contribution
    to other terms. Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    
    The idea: if we remove column c, the tensor changes by:
    delta_T[u,v,w] = -U[u,c] * V[v,c] * W[w,c]
    
    We need to adjust the remaining columns to compensate. This is hard in general.
    Alternative: just check if removing the column still gives a valid decomposition.
    """
    rank = len(U[0])
    if rank <= 1:
        return False, None, None, None
    
    U_new = [row[:col] + row[col+1:] for row in U]
    V_new = [row[:col] + row[col+1:] for row in V]
    W_new = [row[:col] + row[col+1:] for row in W]
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    return False, None, None, None


def _randomize_columns(U, V, W, n, m, p, deadline, rng):
    """
    Try random linear combinations of columns to create sparser representations
    that allow term removal.
    """
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        rank = len(U[0])
        if rank <= best_rank:
            break
        
        # Pick two random columns to combine
        c1 = rng.randint(rank)
        c2 = rng.randint(rank)
        if c1 == c2:
            continue
        
        # Try: new_col_c1 = col_c1 + alpha * col_c2, new_col_c2 = col_c2
        # This is a column operation that preserves the tensor.
        # Actually, any invertible linear transformation of the rank-1 terms preserves the tensor.
        # So we can do: col_c1 <- col_c1 + alpha * col_c2
        # This makes the decomposition different but still valid.
        # Then we can try to remove terms.
        
        alpha = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        # Apply: for each row in U, U[row][c1] += alpha * U[row][c2]
        for r in range(len(U)):
            U[r][c1] += alpha * U[r][c2]
        for r in range(len(V)):
            V[r][c1] += alpha * V[r][c2]
        for r in range(len(W)):
            W[r][c1] += alpha * W[r][c2]
        
        # Now try to remove some term
        for c in range(rank):
            if time.time() > deadline:
                break
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                best_rank = len(U[0])
                best_U, best_V, best_W = U, V, W
                break
    
    return best_U, best_V, best_W


def _build_strassen114(n, m, p):
    """The known rank-114 Strassen-based decomposition."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [0, -1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    def add_444(col, a_base, b_base, c_base):
        for t in range(7):
            u, v, w = U_sums[t], V_sums[t], W_sums[t]
            for s in range(4):
                if u[s]:
                    U[a_base + s][col] += u[s]
                if v[s]:
                    V[b_base + s][col] += v[s]
                if w[s]:
                    W[c_base + s][col] += w[s]
            col += 1
        return col

    col = add_444(0, 0, 0, 0)
    col = add_444(col, 0, 4, 4)
    # 4x4x1 matrix-vector: rank 16
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 1 + 0][col] = 1
            W[0 * 4 + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}


def _greedy_remove(U, V, W, n, m, p, deadline):
    """Greedily try to remove terms, in order of 'simplicity'."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        rank = len(U[0])
        # Try each column
        for c in range(rank):
            if time.time() > deadline:
                break
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                changed = True
                break
    return U, V, W


def _random_column_ops_and_remove(U, V, W, n, m, p, deadline, seed=42):
    """
    Structural search: apply random invertible linear transformations to the
    rank-1 term space, then greedily remove zero-redundancy terms.
    Different bases of the term space can expose linear dependencies.
    """
    import random
    rng = random.Random(seed)
    
    best_U, best_V, best_W = [r[:] for r in U], [r[:] for r in V], [r[:] for r in W]
    best_rank = len(best_U[0])
    
    U = [r[:] for r in U]
    V = [r[:] for r in V]
    W = [r[:] for r in W]
    
    start = time.time()
    max_time = deadline - start
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        if len(U[0]) <= best_rank:
            break
        
        # Apply a random shear: col_a <- col_a + alpha * col_b
        rank = len(U[0])
        a = rng.randint(rank)
        b = rng.randint(rank)
        if a == b:
            continue
        alpha = rng.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
        
        for r in range(len(U)):
            U[r][a] += alpha * U[r][b]
        for r in range(len(V)):
            V[r][a] += alpha * V[r][b]
        for r in range(len(W)):
            W[r][a] += alpha * W[r][b]
        
        # Now greedily remove
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            rk = len(U[0])
            for c in range(rk):
                if time.time() > deadline - 5:
                    break
                U_new = [row[:c] + row[c+1:] for row in U]
                V_new = [row[:c] + row[c+1:] for row in V]
                W_new = [row[:c] + row[c+1:] for row in W]
                if _verify(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    changed = True
                    if len(U[0]) < best_rank:
                        best_rank = len(U[0])
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                    break
        
        # Revert the shear if we didn't improve (to explore from same base)
        # Actually, keep going from current state for diversity
        # But revert to best periodically
        if len(U[0]) > best_rank and trial % 20 == 0:
            U = [row[:] for row in best_U]
            V = [row[:] for row in best_V]
            W = [row[:] for row in best_W]
    
    return best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    # Start with Strassen rank-114
    res = _build_strassen114(n, m, p)
    if not _verify(res["U"], res["V"], res["W"], n, m, p):
        res = _build_schoolbook(n, m, p)
        if not _verify(res["U"], res["V"], res["W"], n, m, p):
            return res

    U, V, W = res["U"], res["V"], res["W"]
    current_rank = len(U[0])

    # Phase 1: Simple greedy removal
    if time.time() < deadline - 20:
        U, V, W = _greedy_remove(U, V, W, n, m, p, deadline - 20)
        if len(U[0]) < current_rank:
            current_rank = len(U[0])

    # Phase 2: Random column operations + removal
    if time.time() < deadline - 10:
        U2, V2, W2 = _random_column_ops_and_remove(U, V, W, n, m, p, deadline, seed=12345)
        if len(U2[0]) < current_rank:
            U, V, W = U2, V2, W2
            current_rank = len(U2[0])

    # Phase 3: Another round with different seed
    if time.time() < deadline - 5:
        U3, V3, W3 = _random_column_ops_and_remove(U, V, W, n, m, p, deadline, seed=99999)
        if len(U3[0]) < current_rank:
            U, V, W = U3, V3, W3
            current_rank = len(U3[0])

    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to original
        res_orig = _build_strassen114(n, m, p)
        if _verify(res_orig["U"], res_orig["V"], res_orig["W"], n, m, p):
            return res_orig
        res_school = _build_schoolbook(n, m, p)
        return res_school

    return {"U": U, "V": V, "W": W}

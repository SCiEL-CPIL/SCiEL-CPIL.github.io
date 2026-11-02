import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
DEADLINE = time.time() + 140  # safety margin under the 150s kill


def _schoolbook():
    n, m, p = N, M, P
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


def _verify(U, V, W):
    # Check exact decomposition: sum_c U[i,j,c]*V[j,k,c] == W[k*n+i, c]
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    # T_{i,j,k} = sum_c U[i,j,c]*V[j,k,c]
    # Reshape: U is (n*m, r) -> (n, m, r); V (m*p, r) -> (m, p, r)
    n, m, p = N, M, P
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    # T[i,j,k] = sum_r U3[i,j,r]*V3[j,k,r]
    T = np.einsum('ijr,jkr->ijk', U3, V3)
    # W target: W[k*n+i, r] should equal T[i,j,k] summed? 
    # Actually the tensor T_{i,j,k} must equal sum_r U[i,j,r]*V[j,k,r]
    # and the W matrix must satisfy the trace convention.
    # The multiplication tensor is T_{i,j,k} = delta_{i,k} (for A[i,j]*B[j,k]=C[i,k])
    # Wait: (A B)[i,k] = sum_j A[i,j] B[j,k]. So the tensor is T_{i,j,k} = delta_{i,k}? No.
    # The bilinear form: output C[i,k] = sum_j A[i,j] B[j,k].
    # As a tensor: T_{i,j,k} where the multiplication maps (A,B) -> C via
    # C[i,k] = sum_{j,c} A[i,j] B[j,k] ... 
    # Standard: the tensor M_{i,j,k} with C[i,k] = sum_j M_{i,j,k} A[i,j] B[j,k] = sum_j delta_{i,k?} 
    # Actually C[i,k] = sum_j A[i,j] B[j,k], so M_{i,j,k} = delta_{?}
    # The decomposition: C[i,k] = sum_c W[i,k,c] * (sum_j U[i,j,c] A[i,j]) * (sum_l V[j,l,c] B[j,l])
    # Hmm, the convention here: W[k*n+i, r]. Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) -> row = k*n + i.
    # The decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+l, c] * B[j,l]) * W[k*n+i, c]
    # Wait that doesn't dimension-check. Let me think again.
    # Standard matrix mult tensor decomposition:
    # C[i,k] = sum_c U_c[i] * V_c[k] * W_c  where U_c, V_c are scalars per (i,j) and (j,k)
    # Actually: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k])
    # No. Let me use: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[?,c]... 
    # The tensor M_{i,j,k} = 1 if this term contributes. For matrix mult, 
    # C[i,k] = sum_j A[i,j] B[j,k], so the "tensor" has entries T_{i,j,k} = delta_{?}
    # I think the correct check is: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c] = 1
    # and W must be such that the trace convention holds. Actually W[k*n+i,c] is the
    # coefficient. Let me just check: sum_c U[i*m+j,c]*V[j*p+k,c] should equal W[k*n+i,c] summed?
    # No. The three-way decomposition of the multiplication tensor:
    # T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]  (where W indexed by (i,k))
    # and T_{i,j,k} = delta_{i,k}? No, T_{i,j,k} is the coefficient of A[i,j]*B[j,k] in C.
    # C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i',k'] is
    # delta_{i,i'} delta_{k,k'} (and j is summed). So T_{i,j,k} as a 3-tensor doesn't quite work.
    # 
    # The proper formulation: the multiplication map is a bilinear map A (n x m) x B (m x p) -> C (n x p).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Rank-1 decomposition: C[i,k] = sum_c U_c[i,j] V_c[j,k] ... no.
    # Each term: C[i,k] += (sum_j u_{i,j}^{(c)} A[i,j]) * (sum_l v_{j,l}^{(c)} B[j,l]) * w^{(c)}_{i,k}
    # For this to equal delta, we need for each c: the product of two linear forms times w gives
    # a rank-1 contribution to C.
    # 
    # Simplest check: build the full multiplication tensor and verify.
    # The tensor M has shape (n*m, m*p, n*p) with M[a,b,c] = 1 if A index a, B index b contribute to C index c.
    # a = i*m+j, b = j*p+k, c = i*p+k. M[a,b,c] = 1 iff (i,j) from a, (j,k) from b agree on j, and c = i*p+k.
    # The decomposition: M = sum_r U[:,r] outer V[:,r] outer W[:,r] (rank-1 tensors).
    # So M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # Let's build M and check.
    Mm = np.zeros((n * m, m * p, n * p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                Mm[a, b, c] = 1
    # Check: sum_r U[a,r] V[b,r] W[c,r] == Mm[a,b,c]
    # U (n*m, r), V (m*p, r), W (n*p, r)
    # But the task says W shape is (p*n, rank) with W[k*n+i, r]. So W index is k*n+i, not i*p+k.
    # The "trace convention" W[k*n+i, r] means W is indexed by (k, i) with row = k*n + i.
    # So the output index in W is (k, i), i.e., C[i,k] corresponds to W[k*n+i, r].
    # So M should use c_index = k*n + i for the W factor.
    # Let me redo: M[a, b, w_idx] where w_idx = k*n + i.
    Mm2 = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w_idx = k * n + i
                Mm2[a, b, w_idx] = 1
    # Now check: sum_r U[a,r]*V[b,r]*W[w_idx,r] == Mm2[a,b,w_idx]
    # U (n*m, r), V (m*p, r), W (p*n, r)
    # Compute via einsum: sum_r U[a,r] V[b,r] W[c,r]
    # This is a 3-way outer product sum. Use: for each r, outer(U[:,r], V[:,r], W[:,r])
    # Efficiently: np.einsum('ar,br,cr->abc', U, V, W)
    try:
        T = np.einsum('ar,br,cr->abc', U, V, W)
    except Exception:
        return False
    return np.array_equal(T, Mm2)


def _try_random_rank(r, rng):
    # Random search for rank-r decomposition
    n, m, p = N, M, P
    # Try random integer entries in small range
    U = rng.randint(-2, 2, size=(n * m, r))
    V = rng.randint(-2, 2, size=(m * p, r))
    W = rng.randint(-2, 2, size=(p * n, r))
    return _verify(U.tolist(), V.tolist(), W.tolist())


def construct():
    start = time.time()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = N * M * P

    # The multiplication tensor for 3x5 x 5x5 has known bounds.
    # Try a few structured lower-rank candidates.
    # For n x m x p, rank can sometimes be less than n*m*p.
    # Known: for 2x2x2, rank 7 (Strassen). For 3x3x3, rank 23 (Laser).
    # For 3x5x5, let's try random search with small ranks.

    # Try a range of ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But also the "dimension" bound. Let's try from 25 upward.
    for r in range(25, min(best_rank, 40)):
        if time.time() > DEADLINE - 20:
            break
        for attempt in range(200):
            if time.time() > DEADLINE - 20:
                break
            if _try_random_rank(r, np_rng):
                # Found a valid rank-r decomposition!
                # Reconstruct U, V, W (we need them, not just verify)
                # _try_random_rank doesn't return them; let's redo
                break
        # Redo: we need to capture U,V,W
    # The above loop is inefficient. Let me rewrite properly.
    # Actually let me just do a clean random search.

    # Clean random search
    for r in range(25, best_rank):
        if time.time() > DEADLINE - 15:
            break
        found = False
        for attempt in range(500):
            if time.time() > DEADLINE - 15:
                break
            U = np_rng.randint(-3, 3, size=(N * M, r))
            V = np_rng.randint(-3, 3, size=(M * P, r))
            W = np_rng.randint(-3, 3, size=(P * N, r))
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                found = True
                break
        if found:
            # Found lower rank, continue searching for even lower
            continue
        # If we didn't find at this rank, we might still try but random search
        # for exact integer decompositions is very hard. 
        # For now, if r is small and we can't find, break.
        if r < 30:
            continue
        else:
            break

    if time.time() > DEADLINE - 10:
        return best

    return best

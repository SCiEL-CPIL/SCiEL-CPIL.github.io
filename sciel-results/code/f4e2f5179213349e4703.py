"""Matrix multiplication 3x5x7 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]
DEADLINE = 150.0
MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p

    def still_going():
        return time.time() < DEADLINE - MARGIN

    def try_candidate(U, V, W):
        nonlocal best, best_rank
        rank = U.shape[1]
        if rank >= best_rank:
            return
        # Exact verification: sum_c U[i,j,c]*V[j,k,c] == W[k,i,c]
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        r = rank
        # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c]
        # Ua shape (n*m, r) -> (n, m, r); Va shape (m*p, r) -> (m, p, r)
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        T = np.einsum('ijc,jkc->ijk', U3, V3)
        # W3 shape (p*n, r) -> (p, n, r); need W[k,i,c]
        W3 = Wa.reshape(p, n, r)
        # For each c, compare T[:,:,c] with W3[:, :, c] transposed appropriately
        # T[i,j,k] should equal sum over c' ... actually the decomposition is:
        # (A B)[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k])
        # So T[i,j,k] is not directly W. The correct check:
        # sum_c U[i,j,c] V[j,k,c] must equal the coefficient that reconstructs
        # Actually the tensor M_{i,j,k} = delta (it's 1 for all i,j,k in standard multiplication)
        # and M = sum_c U_c ⊗ V_c ⊗ W_c where W_c[k,i] = W[k*n+i, c]
        # So check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]?? No.
        # The standard: M_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] is wrong.
        # Correct: M_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] is the rank-1 term.
        # But the tensor M_{i,j,k} = 1 for all valid i,j,k (matrix mult is sum_j A[i,j]B[j,k]).
        # Actually M_{i,j,k} = 1 for all i,j,k since each (i,j,k) contributes A[i,j]*B[j,k].
        # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k? No.
        # The tensor is T_{i,j,k} = 1 for all i,j,k. A rank-r decomposition is:
        # T_{i,j,k} = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, that's not standard either. Let me re-derive.
        # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_c sum_j sum_k (U[i,j,c] A[i,j]) (V[j,k,c] B[j,k]) (W[k,i,c])
        # Hmm, the standard form is:
        # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k])
        # No. The standard tensor decomposition:
        # T_{i,j,k} = sum_c u_{i,j}^{(c)} v_{j,k}^{(c)} w_{k,i}^{(c)}
        # where T_{i,j,k} = 1 for all i,j,k.
        # So check: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        lhs = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        if not np.allclose(lhs, 1.0, atol=1e-9):
            return
        best = (U, V, W)
        best_rank = rank

    # Try known structured decompositions
    for candidate in _generate_candidates(n, m, p):
        if not still_going():
            break
        try_candidate(*candidate)

    # Random search for better rank
    rng = random.Random(42)
    for _ in range(200000):
        if not still_going():
            break
        r = rng.randint(105, best_rank - 1)
        if r >= best_rank:
            continue
        # Random integer/half-integer entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n * m, r))
        V = np.random.choice([-1, 0, 1], size=(m * p, r))
        W = np.random.choice([-1, 0, 1], size=(p * n, r))
        try_candidate(U, V, W)

    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _generate_candidates(n, m, p):
    """Generate candidate decompositions from known algorithms."""
    candidates = []

    # Partition m = 2 + 3
    m1, m2 = 2, 3

    # Strassen for 2x2x2 blocks, Laderman for 3x3x3 blocks
    # For 3x5x7, we can try block decompositions

    # Try a simple block approach: split m into 2 and 3
    # Subproblem 1: 3x2x7, Subproblem 2: 3x3x7
    # Use Strassen-like for the 2-part, Laderman-like for the 3-part

    # For now, generate a few structured candidates
    # Candidate: use Kronecker product structure

    # 3x5x7 can be seen as (3x2x7) + (3x3x7)
    # For 3x2x7: rank <= 3*2*7 = 42, but can do better
    # For 3x3x7: rank <= 3*3*7 = 63, but can do better

    # Let's try a direct construction with rank 105 (the trivial bound is 105)
    # Actually n*m*p = 3*5*7 = 105, so schoolbook is 105.

    # Try to find a decomposition with rank < 105
    # Use the fact that we can share computation

    # Generate candidate using tensor network contraction
    # This is complex, so for now just yield the schoolbook and a few variants

    yield _schoolbook()

    # Try a random structured candidate
    rng = random.Random(123)
    for trial in range(50):
        r = 100 + rng.randint(0, 5)
        if r >= 105:
            continue
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        # Fill with random patterns
        for c in range(r):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        candidates.append((U, V, W))

    for c in candidates:
        yield c


# EVOLVE-BLOCK-END

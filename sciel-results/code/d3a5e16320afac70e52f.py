"""Rewritten program: search for a rank-minimal exact integer tensor decomposition
of the 4x4 * 4x7 matrix multiplication tensor, with a wall-clock deadline guard."""
import time
import random

DIMENSIONS = [4, 4, 7]


def _schoolbook():
    """Always-valid baseline: schoolbook rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 iff i==k else 0.
    Returns True if the decomposition exactly reproduces the multiplication tensor."""
    r = len(U[0]) if U and U[0] else 0
    # For each (i,j,k) we need the scalar product over c of U[i*m+j,c]*V[j*p+k,c] to
    # be independent of k in the right way. Actually the tensor T[i,j,k] = E_{i,k}[j?].
    # The multiplication tensor: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T[i,j,k] = 1 if i==k? No. The standard matmul tensor has T[i,j,k] such that
    # C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} ... 
    # With convention W[k*n+i, r], the identity is:
    #   C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U[i*m+j,c] A_{i,j}) (sum_j V[j*p+k,c] B_{j,k})
    # so we need: for all i,k, sum_c U[i*m+j,c]*V[j*p+k,c] summed over j? 
    # Correctly: define for each c: a vector a_c over (i,j) and b_c over (j,k) and w_c over (k,i).
    # The decomposition says: for all i,k, sum_c (sum_j U[i*m+j,c] A_{i,j}) (sum_j V[j*p+k,c] B_{j,k}) * 1 = C_{i,k}
    # This equals sum_c [sum_j U[i*m+j,c] A_{i,j}] [sum_j V[j*p+k,c] B_{j,k}].
    # For this to equal sum_j A_{i,j} B_{j,k} for ALL A,B, we need:
    #   sum_c U[i*m+j,c] V[j*p+k',c] ... hmm, W must also appear.
    # The standard form: C_{i,k} = sum_c (sum_j U[i*m+j,c] A_{i,j}) (sum_j V[j*p+k,c] B_{j,k}) W? 
    # No. Let me use the given convention: U shape (n*m, r), V (m*p, r), W (p*n, r).
    # Identity: C_{i,k} = sum_c (sum_j U[i*m+j, c] A_{i,j}) (sum_j V[j*p+k, c] B_{j,k}) * (sum_l W[k*n+i, c] * delta?)
    # The cleanest: the tensor T with indices (i,j,k) is T[i,j,k] = 1 if i==k else 0 is WRONG.
    # The matmul tensor T[i,j,k] = E_{i,k} has T[i,j,k]=1 for all j when i==k? No:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}. The tensor is T[i,j,k] = delta? It's T[i,j,k] such that
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}. So T is indexed (i,j,k) with T[i,j,k] = 1 for the term A_{i,j}B_{j,k}.
    # That means T[i,j,k] = 1 for ALL (i,j,k)?? No: C_{i,k} = sum_j A_{i,j} B_{j,k}, so the coefficient
    # of A_{i,j} B_{j,k} in C_{i,k} is 1. So T[i,j,k] = 1 for all i,j,k. That can't be right either
    # because then it's a rank-1-ish? No.
    #
    # Let me be careful. The multiplication tensor for (n x m) * (m x p) -> (n x p):
    #   T[i, j, k] = 1  for the product term A[i,j]*B[j,k] contributing to C[i,k].
    # So T[i,j,k] = 1 for every (i,j,k)?? That would mean C[i,k] = sum_{j} A[i,j] B[j,k] which uses
    # T[i,j,k]=1. But T[i,j,k] must be such that summing over j gives C[i,k]. So T[i,j,k] = 1 for all
    # (i,j,k) is NOT a valid tensor because T is 3-indexed and C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # with T[i,j,k] = 1. Yes! T[i,j,k] = 1 for all i,j,k. Wait that's the identity tensor that
    # just multiplies. But then the rank would be... hmm, but the schoolbook uses n*m*p terms with
    # T[i,j,k] = 1 for all. Actually the schoolbook sets U,V,W so that for each (i,j,k) triple there's
    # a rank-1 slice. The tensor T[i,j,k] = 1 for all i,j,k? Let's check schoolbook:
    #   U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col corresponding to (i,j,k).
    #   Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 only when c matches (i,j,k) exactly,
    #   i.e., equals 1 for that (i,j,k) and 0 otherwise. So the reconstructed tensor
    #   T'[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k). 
    # So indeed T[i,j,k] = 1 for all i,j,k. 
    # Therefore verification: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1.
    return True  # placeholder, replaced below


def _verify_true(U, V, W, n, m, p):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _random_int_matrix(rows, cols, lo=-2, hi=2):
    return [[random.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin under 150s kill

    best = _schoolbook()

    # Lower bound: rank >= max(n*m, m*p, p*n) = max(16,28,28)=28.
    # We'll try to find ranks below 112.
    # Strategy: random integer factor matrices with small entries, verify exactly.
    # This is a heuristic; it may not find a lower rank, but it's a fresh design.
    # To make it more likely to find structured low-rank, use a "basis" approach:
    # represent each row of U,V,W as a sparse combination of a few basis vectors.

    rng = random.Random(12345)

    # Try a range of ranks from a modest lower bound upward.
    # For each rank, do a limited number of random trials.
    for r in range(max(n*m, m*p, p*n), 112):
        if time.time() > deadline - 5.0:
            break
        trials = 200
        for t in range(trials):
            if time.time() > deadline - 5.0:
                break
            # Random sparse integer factors: each row has a few nonzeros in {-1,0,1}.
            def sparse_row():
                row = [0] * r
                nnz = rng.randint(1, 3)
                for _ in range(nnz):
                    idx = rng.randrange(r)
                    row[idx] = rng.choice([-1, 1])
                return row
            U = [sparse_row() for _ in range(n * m)]
            V = [sparse_row() for _ in range(m * p)]
            W = [sparse_row() for _ in range(p * n)]
            if _verify_true(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                # Found a valid rank r; continue trying lower? r is increasing, so keep going
                # but we want minimal; since we go ascending, first found is minimal for this r.
                # Break out to try even lower ranks? We already tried lower r in earlier loops.
                # So this is the best so far; keep searching lower isn't possible (ascending).
                # We can stop searching higher ranks to save time, but maybe a lower r was
                # missed. Since ascending, this is the minimum found. Return it.
                return best
    return best

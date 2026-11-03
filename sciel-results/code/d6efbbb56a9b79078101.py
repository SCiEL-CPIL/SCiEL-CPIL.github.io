"""Rewritten program: structured Strassen-like decomposition for 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Start with schoolbook (rank 54)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank, n, m, p, tol=1e-9):
        """Verify exact decomposition using numpy for speed."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # For each (i,j,k) check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Vectorized: for each c, the contribution to (i,j,k) is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We need the sum over c to be 1 for all (i,j,k)
        # Build the tensor implicitly
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can compute this by iterating over c and accumulating
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c]  # shape (n*m,)
            v = V[:, c]  # shape (m*p,)
            w = W[:, c]  # shape (p*n,)
            # u reshaped to (n, m), v to (m, p), w to (p, n)
            u3 = u.reshape(n, m)
            v3 = v.reshape(m, p)
            w3 = w.reshape(p, n)
            # Contribution: T[i,j,k] += u3[i,j] * v3[j,k] * w3[k,i]
            # This is an outer-product-like operation
            # For each (i,j,k): u3[i,j] * v3[j,k] * w3[k,i]
            # We can do: for each i, j: u3[i,j] * (v3[j,:] * w3[:,i])
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u3[i, j] * (v3[j, :] * w3[:, i])
        return np.all(np.abs(T - 1.0) <= tol)

    # ============================================================
    # Strategy 1: Direct structured decomposition using the block
    # structure. C = A * B where A is 3x3, B is 3x6.
    # Split B = [B1 | B2] where B1, B2 are 3x3.
    # C = [A*B1 | A*B2]
    #
    # Key idea: use a rank-r decomposition for the 3x3x3 tensor,
    # then "lift" it to handle both B1 and B2 by duplicating the
    # V factors appropriately.
    #
    # For a 3x3x3 decomposition of rank r:
    #   (A*B)[i,k] = sum_c U[i,k,c] * V[j,k,c] * W[i,j,c]  -- wait,
    #   actually the tensor is T[i,j,k] = A[i,j]*B[j,k] = C[i,k]
    #   So: C[i,k] = sum_c U[i,k,c] * V[j,k,c] * W[i,j,c]
    #   where U has shape (n*p, r) = (3*3, r), V has shape (m*p, r) = (3*3, r),
    #   W has shape (p*n, r) = (3*3, r)... 
    #
    # Wait, let me re-derive. The matrix multiplication tensor for n x m by m x p:
    # T[i, j, k] = A[i,j] * B[j,k] contributes to C[i,k].
    # A rank-r decomposition means:
    #   C[i,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  ... no
    #
    # Actually, the standard formulation:
    # The multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for the (i,j,k)
    # "term" in the schoolbook. A rank-r decomposition gives matrices
    # U (n*m x r), V (m*p x r), W (p*n x r) such that for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is what the verify function checks.
    #
    # For the 3x3x6 case:
    # U: 9 x r, V: 18 x r, W: 18 x r
    #
    # Block structure: split k into k1 in {0,1,2} and k2 in {0,1,2} with k = k1 + 3*k2.
    # V[j*p+k, c] = V[j*6+k, c]. For k = k1 (k2=0): V[j*6+k1, c]
    # For k = 3+k1 (k2=1): V[j*6+3+k1, c]
    #
    # If we could find a 3x3x3 decomposition of rank r with matrices
    # U0 (9 x r), V0 (9 x r), W0 (9 x r) such that
    #   sum_c U0[i*3+j, c] * V0[j*3+k, c] * W0[k*3+i, c] = 1 for all i,j,k in {0,1,2}
    #
    # Then for the 3x3x6 case, we could set:
    #   U[i*3+j, c] = U0[i*3+j, c]  (same for both blocks)
    #   V[j*6+k1, c] = V0[j*3+k1, c]  for k2=0
    #   V[j*6+3+k1, c] = V0[j*3+k1, c]  for k2=1
    #   W[k1*3+i, c] = W0[k1*3+i, c]  for k2=0
    #   W[(3+k1)*3+i, c] = W0[k1*3+i, c]  for k2=1
    #
    # But this would give rank r, not 2r! Because we're reusing the same c indices.
    # However, this only works if the same c can serve both blocks, which means
    # the V and W entries for the two blocks must be consistent.
    #
    # Actually, this approach would give a valid 3x3x6 decomposition of rank r
    # (same as the 3x3x3 rank), which would be amazing but is impossible because
    # the 3x3x6 tensor has more "independent" structure.
    #
    # Let me think differently. The 3x3x6 tensor can be seen as:
    #   C[i, k] = sum_j A[i,j] * B[j,k]
    # For k in 0,1,2: C[i,k] = sum_j A[i,j] * B[j,k]
    # For k in 3,4,5: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # These are two separate 3x3x3 multiplications that share the same A.
    # So the tensor is the "direct sum" in a sense, but they share the A indices.
    #
    # A valid decomposition of rank 2r (where r is the 3x3x3 rank) would be:
    #   Use r terms for the first block and r terms for the second block.
    #   For c in 0..r-1: handles k in 0,1,2
    #   For c in r..2r-1: handles k in 3,4,5
    #
    # This gives rank 2*r(3x3x3). If r(3x3x3) = 23 (Strassen), then rank = 46.
    #
    # But can we do better by sharing some terms between the two blocks?
    # The key constraint is that U[i*m+j, c] must be the same for both blocks
    # (since it depends only on i and j). But V and W depend on k, so they can
    # differ between blocks.
    #
    # Actually, U depends on (i,j) which is the same for both blocks. So if we
    # use the same U for both blocks, we need:
    #   For k in 0,1,2: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #   For k in 3,4,5: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # If we split c into two groups, we get rank 2r. But if we allow the same c
    # to contribute to both blocks, we might do better.
    #
    # Let's try a different approach: use a rank-r 3x3x3 decomposition and
    # "extend" it to 3x3x6 by allowing V and W to have different values for
    # the two k-blocks.
    #
    # Actually, the cleanest approach:
    # 1. Find a rank-r decomposition for 3x3x3.
    # 2. For 3x3x6, use 2r terms: r for each block.
    # 3. Try to reduce below 2r by merging terms.

    # ============================================================
    # Strategy 2: Random search with structured initialization
    # ============================================================

    # First, try to find a good 3x3x3 decomposition
    def find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=60):
        """Find a rank-r decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        best = None
        best_r = max_rank + 1

        def verify_3x3x3(U, V, W, r):
            U = np.array(U, dtype=np.float64)
            V = np.array(V, dtype=np.float64)
            W = np.array(W, dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True

        for r in range(max_rank, min_rank - 1, -1):
            if time.time() > start_time + max_time:
                break
            for attempt in range(20):
                if time.time() > start_time + max_time:
                    break
                rng = np.random.RandomState(attempt * 7919 + r * 31)
                vals = [-1, 0, 1]
                U = rng.choice(vals, size=(9, r))
                V = rng.choice(vals, size=(9, r))
                W = rng.choice(vals, size=(9, r))
                if verify_3x3x3(U, V, W, r):
                    best = (U, V, W, r)
                    best_r = r
                    break
        return best

    # Try to find 3x3x3 decomposition
    t3 = time.time()
    dec3 = find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=30)

    if dec3 is not None:
        U3, V3, W3, r3 = dec3
        # Build 3x3x6 decomposition from 3x3x3
        # Use 2*r3 terms: r3 for block 1 (k=0,1,2), r3 for block 2 (k=3,4,5)
        r6 = 2 * r3
        U6 = np.zeros((9, r6))
        V6 = np.zeros((18, r6))
        W6 = np.zeros((18, r6))

        # Block 1: k in 0,1,2
        U6[:, :r3] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+k, :r3] = V3[j*3+k]
                W6[k*3+i, :r3] = W3[k*3+i]  # Wait, this is wrong, W depends on k and i

        # Let me redo this more carefully
        U6 = np.zeros((9, r6))
        V6 = np.zeros((18, r6))
        W6 = np.zeros((18, r6))

        # Block 1: c in 0..r3-1, k in 0,1,2
        U6[:, :r3] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+k, :r3] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[k*3+i, :r3] = W3[k*3+i]

        # Block 2: c in r3..2*r3-1, k in 3,4,5
        U6[:, r3:] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+3+k, r3:] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[(3+k)*3+i, r3:] = W3[k*3+i]

        if verify_decomposition(U6.tolist(), V6.tolist(), W6.tolist(), r6, n, m, p):
            if r6 < best_rank:
                best_rank = r6
                best_result = {
                    "U": U6.tolist(),
                    "V": V6.tolist(),
                    "W": W6.tolist()
                }

    # ============================================================
    # Strategy 3: Direct random search for 3x3x6
    # ============================================================
    for rank in range(min(50, best_rank - 1), 30, -1):
        if time.time() > deadline - 15:
            break

        max_attempts = 30 if rank > 40 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            rng = np.random.RandomState(attempt * 104729 + rank * 13)
            density = 0.3
            vals = [-1, 0, 1]
            U = rng.choice(vals, size=(9, rank))
            V = rng.choice(vals, size=(18, rank))
            W = rng.choice(vals, size=(18, rank))

            # Apply sparsity
            U = U * (rng.random((9, rank)) < density)
            V = V * (rng.random((18, rank)) < density)
            W = W * (rng.random((18, rank)) < density)

            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break

    # ============================================================
    # Strategy 4: Local search - try to reduce rank from current best
    # ============================================================
    if best_rank < 54 and time.time() < deadline - 20:
        U_best = np.array(best_result["U"], dtype=np.float64)
        V_best = np.array(best_result["V"], dtype=np.float64)
        W_best = np.array(best_result["W"], dtype=np.float64)
        r_best = best_rank

        # Try removing columns (reducing rank)
        for target_rank in range(r_best - 1, max(30, r_best - 10), -1):
            if time.time() > deadline - 10:
                break
            # Try all subsets of columns of size target_rank
            # This is expensive, so try random subsets
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                rng = np.random.RandomState(attempt * 9973 + target_rank)
                cols = rng.choice(r_best, size=target_rank, replace=False)
                cols.sort()
                U_sub = U_best[:, cols]
                V_sub = V_best[:, cols]
                W_sub = W_best[:, cols]
                if verify_decomposition(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), target_rank, n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U_sub.tolist(),
                        "V": V_sub.tolist(),
                        "W": W_sub.tolist()
                    }
                    break

    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
# EVOLVE-BLOCK-END

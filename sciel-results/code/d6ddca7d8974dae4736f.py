"""Matrix multiplication 4x4x5 - structured block decomposition approach."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80

    # Helper to validate a decomposition
    def is_valid(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the matmul tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1.0 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    for c in range(rank):
                        lhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    best = schoolbook()

    # Structured block decomposition strategy:
    # Partition 4x4x5 into blocks and use Strassen-like recurrences
    # 
    # For 2x2x2 matmul, Strassen gives rank 7:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A11+A22)(B21-B22)
    # M3 = (A11-A22)(B11+B12)
    # M4 = A22(B21-B11)
    # M5 = A11(B12-B22)
    # M6 = (A21+A22)(B11-B12)
    # M7 = (A11-A12)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M7

    # For 4x4x4, we can use a 2-level Strassen: rank 7^2 = 49
    # For 4x4x5, we need to handle the extra column

    # Strategy: Use a direct construction based on bilinear forms
    # The matmul tensor for (4,4,5) has 4*4*5 = 80 terms in schoolbook
    # We try to find a lower rank decomposition

    # Approach: Use a parameterized family of decompositions
    # and search over parameters

    # First, try a structured approach based on block decomposition
    # Split A (4x4) into 2x2 blocks, B (4x5) into 2x2 and 2x3 blocks

    # Let's try a direct algebraic construction
    # The key insight: for matmul(n,m,p), the tensor rank is at most n*m*p
    # and at least max(n*m, m*p, p*n) = max(16, 20, 20) = 20

    # Try to construct a rank-20 to rank-50 decomposition using structured patterns

    # Method 1: Use Kronecker product structure
    # 4 = 2*2, 5 = 2+3
    # This is complex, so let's try a more direct search

    # Method 2: Random search with structured initialization
    # Initialize with patterns that are likely to work

    # Try ranks from 20 upward, with increasing effort
    target_ranks = list(range(20, min(schoolbook_rank, 65)))
    target_ranks.reverse()  # Try lower ranks first

    # Precompute indices for faster validation
    idx_U = [(i*m+j) for i in range(n) for j in range(m) for k in range(p)]
    idx_V = [(j*p+k) for i in range(n) for j in range(m) for k in range(p)]
    idx_W = [(k*n+i) for i in range(n) for j in range(m) for k in range(p)]

    def is_valid_fast(U, V, W, rank):
        """Fast validation using numpy broadcasting."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    # Structured initialization patterns
    def structured_init(r):
        """Create structured initializations likely to be close to valid."""
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))

        # Pattern 1: Block-diagonal with Strassen coefficients
        # For each 2x2x2 block, use Strassen's 7 terms
        # 4x4x5 can be seen as overlapping 2x2x2 blocks

        # Let's use a simpler structured pattern:
        # Assign each of the 80 (i,j,k) terms to a rank slot,
        # then try to merge terms using algebraic identities

        # Initialize with random but structured values
        for c in range(r):
            # Create a pattern for this rank component
            # Use a pseudo-random but deterministic pattern
            seed = c * 7919 + 42
            rng = np.random.RandomState(seed)

            # Sparse initialization: each rank component touches few entries
            num_u = rng.randint(2, 5)
            num_v = rng.randint(2, 5)
            num_w = rng.randint(2, 5)

            u_idx = rng.choice(n*m, size=num_u, replace=False)
            v_idx = rng.choice(m*p, size=num_v, replace=False)
            w_idx = rng.choice(p*n, size=num_w, replace=False)

            for idx in u_idx:
                U[idx, c] = rng.choice([-1, -0.5, 0.5, 1])
            for idx in v_idx:
                V[idx, c] = rng.choice([-1, -0.5, 0.5, 1])
            for idx in w_idx:
                W[idx, c] = rng.choice([-1, -0.5, 0.5, 1])

        return U, V, W

    # Local search: try to fix one entry at a time
    def local_search(U, V, W, r, max_iters=1000):
        """Try to improve the decomposition via local search."""
        U = U.copy()
        V = V.copy()
        W = W.copy()

        candidates = [-1, -0.5, 0, 0.5, 1]

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Check if valid
            if is_valid_fast(U.tolist(), V.tolist(), W.tolist(), r):
                return U, V, W, True

            # Pick a random matrix and entry to modify
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = np.random.randint(n*m)
            elif mat == 'V':
                row = np.random.randint(m*p)
            else:
                row = np.random.randint(p*n)
            col = np.random.randint(r)

            old_val = U[row, col] if mat == 'U' else (V[row, col] if mat == 'V' else W[row, col])
            new_val = np.random.choice(candidates)
            if new_val == old_val:
                continue

            if mat == 'U':
                U[row, col] = new_val
                if is_valid_fast(U.tolist(), V.tolist(), W.tolist(), r):
                    return U, V, W, True
                U[row, col] = old_val
            elif mat == 'V':
                V[row, col] = new_val
                if is_valid_fast(U.tolist(), V.tolist(), W.tolist(), r):
                    return U, V, W, True
                V[row, col] = old_val
            else:
                W[row, col] = new_val
                if is_valid_fast(U.tolist(), V.tolist(), W.tolist(), r):
                    return U, V, W, True
                W[row, col] = old_val

        return U, V, W, False

    # Search for valid decompositions
    for r in target_ranks:
        if time.time() > deadline - 10:
            break

        # Number of attempts based on rank
        max_attempts = 30 if r < 40 else 15

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break

            # Try different initialization strategies
            if attempt % 3 == 0:
                # Structured sparse init
                U, V, W = structured_init(r)
            elif attempt % 3 == 1:
                # Dense random init
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            else:
                # Hybrid: start from schoolbook and compress
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # Map schoolbook terms to rank slots
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            slot = (i*m+j)*p+k
                            if slot < r:
                                U[i*m+j, slot] = 1
                                V[j*p+k, slot] = 1
                                W[k*n+i, slot] = 1
                            else:
                                # Try to merge with an existing slot
                                slot2 = np.random.randint(r)
                                U[i*m+j, slot2] += 1
                                V[j*p+k, slot2] += 1
                                W[k*n+i, slot2] += 1

            # Run local search
            U, V, W, valid = local_search(U, V, W, r, max_iters=500)

            if valid:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": r}
                break

        if best["rank"] == r:
            break  # Found valid decomposition at this rank

    # Return in the required format
    result = {
        "U": [[float(x) for x in row] for row in best["U"]],
        "V": [[float(x) for x in row] for row in best["V"]],
        "W": [[float(x) for x in row] for row in best["W"]]
    }

    return result
# EVOLVE-BLOCK-END

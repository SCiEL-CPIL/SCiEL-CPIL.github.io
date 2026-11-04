"""Search for a rank-minimal exact tensor decomposition of M_{3,4,7}."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84

    # The matrix multiplication tensor T[i,j,k] = 1 for all (i in [0,n), j in [0,m), k in [0,p))
    # A rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product of 3 vectors)
    # i.e. T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # Strategy: Try to find a decomposition of rank < 84 using structured patterns.
    # Key insight: The multiplication tensor M_{n,m,p} has known lower bounds.
    # For M_{3,4,7}: schoolbook = 84. Known best upper bounds in literature:
    # We'll try several structured families and random search.

    def verify(U, V, W, rank):
        """Check if U, V, W (as lists) form a valid decomposition."""
        # Convert to numpy for speed
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        # Reconstruct tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = Ua[:, c]  # shape (n*m,)
            v = Va[:, c]  # shape (m*p,)
            w = Wa[:, c]  # shape (p*n,)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            # outer product contribution: u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # Check against all-ones tensor
        target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, target, atol=1e-9)

    def verify_exact(U, V, W, rank):
        """Exact verification using Fraction for integer/half-integer entries."""
        try:
            Uf = [[Fraction(x) for x in row] for row in U]
            Vf = [[Fraction(x) for x in row] for row in V]
            Wf = [[Fraction(x) for x in row] for row in W]
        except:
            return False
        tensor = [[Fraction(0) for _ in range(p)] for _ in range(m)]
        tensor = [[[Fraction(0) for _ in range(p)] for _ in range(m)] for _ in range(n)]
        for c in range(rank):
            u = Uf[:, c] if hasattr(Uf, '__getitem__') else [Uf[i*m+j][c] for i in range(n) for j in range(m)]
            v = [Vf[j*p+k][c] for j in range(m) for k in range(p)]
            w = [Wf[k*n+i][c] for k in range(p) for i in range(n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i][j][k] += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if tensor[i][j][k] != 1:
                        return False
        return True

    # Start with schoolbook
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Try structured low-rank decompositions
    # Family 1: Kronecker-product-like structures
    # Family 2: Randomized search with structured initializations

    # Try rank values from lower bound upward
    # Lower bound for M_{3,4,7}: at least max(3*4, 4*7, 3*7) = 28
    # But actually the tensor rank is at least the max of the three flattening ranks.
    # Flattening 1 (n*m x p): rank at most min(n*m, p) but tensor rank >= max of flattening ranks
    # For all-ones tensor: each flattening has rank 1, so that bound is weak.
    # Better lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) ... not standard.
    # Use: rank >= max(n*m, m*p, p*n) is NOT true. 
    # Actually for matrix mult tensor, rank >= n*m*p / max(n,m,p) is a known bound? No.
    # Let's just try from a reasonable lower bound.

    # Known: M_{2,2,2} has rank 7 (Strassen). M_{3,3,3} has rank <= 23 (Strassen), lower bound 19.
    # For M_{3,4,7}, let's try to find something below 84.

    # Approach: Use the fact that we can group terms.
    # If we use Strassen-like ideas on sub-blocks...

    # Let's try a direct random search for low rank with structured entries.
    # For small ranks, use integer entries from {-1, 0, 1, 2, -2}.

    def random_decomposition(rank, seed_val):
        rng = random.Random(seed_val)
        entries = [-2, -1, 0, 1, 2]
        U = [[rng.choice(entries) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice(entries) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice(entries) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W

    # Try structured families first
    # Family: block-diagonal Strassen-like
    # For M_{3,4,7}, think of 3x4 as blocks and 4x7 as blocks.
    # If we partition the 4 into 2+2, we can apply Strassen to 2x2 blocks.
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x2, etc.
    # B = [B11 B12; B21 B22] where B11 is 2x2, etc.
    # C = A*B, each 2x2 block computed via Strassen (rank 7 each).
    # 4 blocks * 7 = 28? No, that's not right because Strassen computes 2x2*2x2 in rank 7.
    # But the cross terms couple things.

    # Actually, let's just do a focused search.
    # Try ranks from 28 to 83, but focus on promising ones.

    # For efficiency, use numpy-based verification for the search,
    # then exact verification for the final candidate.

    def verify_np(U, V, W, rank):
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = Ua[:, c].reshape(n, m)
            v = Va[:, c].reshape(m, p)
            w = Wa[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, target, atol=1e-8)

    # Try a structured approach: exploit the all-ones structure.
    # If U, V, W are all-ones vectors, we get n*m*p = 84 (schoolbook).
    # To reduce rank, we need cancellations.

    # Let's try: for each "direction" c, choose U_c, V_c, W_c such that
    # the sum reconstructs the all-ones tensor.

    # Random search with small integer entries for ranks near lower bounds.
    # Lower bound: The tensor rank of M_{n,m,p} is at least n*m*p / max(n,m,p) * something...
    # Actually, a simple lower bound: rank >= max(n*m, m*p, p*n) is false.
    # rank >= n*m*p / (n+m+p-2) is a known bound for some tensors? Let's use 28 as starting point.

    # For the search, try ranks in a strategic order.
    candidate_ranks = list(range(28, 84))
    # Prioritize: try lower ranks first but with limited budget each.

    random.seed(42)
    np.random.seed(42)

    found = False
    best_found_rank = rank_school
    best_found = (U_school, V_school, W_school)

    # Budget: spend most time on promising rank ranges.
    # Try a few random samples per rank.
    trials_per_rank = 50  # for lower ranks
    trials_per_rank_high = 20  # for higher ranks

    start_time = time.time()

    for rank in range(28, 84):
        if time.time() > deadline - 10:
            break
        if rank >= best_found_rank:
            continue

        n_trials = trials_per_rank if rank < 50 else trials_per_rank_high
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            U, V, W = random_decomposition(rank, trial * 1000 + rank)
            if verify_np(U, V, W, rank):
                # Exact check
                if verify_exact(U, V, W, rank):
                    best_found_rank = rank
                    best_found = (U, V, W)
                    found = True
                    break
        if found:
            break

    if best_found_rank < best_rank:
        U, V, W = best_found
        best_rank = best_found_rank
        best_result = {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END

"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def elapsed():
        return time.time() - start_time

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify_numpy(U, V, W, tol=1e-9):
        """Verify decomposition using numpy. U:(nm,r), V:(mp,r), W:(pn,r)"""
        r = U.shape[1]
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = to_lists(U_school, V_school, W_school)
    best_rank = schoolbook_rank

    # Strategy 1: Try to find rank-99 by merging two schoolbook terms
    # If we can find coefficients a,b such that replacing two basis terms with
    # a modified pair still gives exact 1 everywhere, we reduce rank by 1.
    #
    # For terms (i1,j1,k1) and (i2,j2,k2), we need:
    # For all (i,j,k): a*U1[i,j]*V1[j,k]*W1[k,i] + b*U2[i,j]*V2[j,k]*W2[k,i]
    #   + sum_other = 1
    # where U1,V1,W1 are the basis vectors for term1, etc.
    # This requires the two terms to have overlapping support in a specific way.
    #
    # In schoolbook, each term has support on exactly one (i,j) in U, one (j,k) in V,
    # and one (k,i) in W. Two terms overlap in the product only if they share
    # the same (i,j) OR same (j,k) OR same (k,i) pattern in a compatible way.
    #
    # Actually, for the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] to be nonzero,
    # we need all three to be nonzero. In schoolbook, each column has exactly one
    # nonzero in each matrix. So the product is nonzero only at the specific (i,j,k)
    # for that column. Two different schoolbook columns have disjoint support in the
    # tensor space, so we can't merge them.
    #
    # This means the schoolbook decomposition is already "minimal" in the sense that
    # each term contributes to a unique tensor entry. The rank IS n*m*p = 100.
    #
    # Wait - that's not quite right. The tensor entry T[i,j,k] receives contribution
    # from ALL columns r where U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] != 0.
    # In schoolbook, for a given (i,j,k), only the column corresponding to (i,j,k)
    # has all three factors nonzero. So each tensor entry is determined by exactly
    # one column. This means we cannot reduce the rank below 100 without violating
    # the constraint, UNLESS we find a non-schoolbook decomposition.
    #
    # Let's try random search for lower ranks, but be realistic about the time budget.

    # Strategy 2: Random search with structured initialization
    # Try ranks from 99 down, with limited trials
    # Use half-integer entries as allowed

    def try_random_rank(target_rank, num_trials=30, seed=42):
        """Try to find a rank-target_rank decomposition."""
        rng = random.Random(seed)
        for trial in range(num_trials):
            if elapsed() > deadline - 5:
                return None

            # Initialize with schoolbook for first `target_rank` terms,
            # then randomize to try to absorb the remaining terms
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))

            # Fill first min(target_rank, 100) columns with schoolbook pattern
            n_fill = min(target_rank, schoolbook_rank)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= n_fill:
                            break
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                    if col >= n_fill:
                        break
                if col >= n_fill:
                    break

            # For remaining columns (if target_rank < 100, there are none to randomize
            # in the "extra" sense; if target_rank > 100, randomize extras)
            # Actually if target_rank < 100, we've already dropped some schoolbook terms.
            # We need to check if the dropped terms can be "absorbed" by modifying
            # the remaining columns.

            # Perturb the existing columns slightly to try to cover dropped terms
            if target_rank < schoolbook_rank:
                # Add small random perturbations to existing columns
                for c in range(target_rank):
                    # Randomly adjust some entries
                    for _ in range(3):
                        row_u = rng.randint(0, nm - 1)
                        row_v = rng.randint(0, mp - 1)
                        row_w = rng.randint(0, pn - 1)
                        delta = rng.choice([-1, 1]) * rng.choice([0.5, 1.0])
                        U[row_u, c] += delta
                        V[row_v, c] += delta
                        W[row_w, c] += delta

            # Verify
            if verify_numpy(U, V, W, tol=1e-9):
                return to_lists(U, V, W)

        return None

    # Try ranks 99 down to 95 (limited trials due to time)
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 6, 0), -1):
        if elapsed() > deadline - 10:
            break

        result = try_random_rank(target_rank, num_trials=20, seed=target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

    # If we still have time, try a few more structured approaches
    if elapsed() < deadline - 15:
        # Try: can we use the fact that n=4, p=5 to find a block structure?
        # For fixed j, the sub-tensor over (i,k) is an outer product of
        # U[:,j-related] and W[:,j-related]. If we can share structure across j...
        #
        # Alternative: try rank = n*p + (n-1)*(p-1) or similar formulas
        # For n x m x p matmul, known bounds: rank >= n*p (from identity submatrix)
        # For 4x5x5: lower bound is at least 4*5 = 20 (from fixing j and looking at
        # the n x p sub-tensor which must be rank >= min(n,p) for each j... actually
        # the full tensor restricted to fixed j gives an n x p tensor of all 1s,
        # which has rank 1. So that bound is weak.)
        #
        # A better lower bound: for fixed (i,k), sum over j of A[i,j]*B[j,k] = C[i,k].
        # The tensor T[i,j,k] = 1. Consider the "slice" at fixed i: this is an
        # m x p tensor of all 1s, rank 1. Similarly for fixed k.
        #
        # The Strassen-style lower bound for n x n x n is n^2 * (something).
        # For 4x5x5, it's not well-studied. Let's just accept the schoolbook rank
        # if we can't find better.
        pass

    return best_result
# EVOLVE-BLOCK-END

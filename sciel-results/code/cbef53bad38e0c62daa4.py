"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook = n * m * p  # 42

    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want T = sum_{r=1}^R U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    def _schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = schoolbook
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s = s + U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        return True

    def try_rank_reduction(target_rank, max_attempts=50):
        """
        Try to find a rank-target_rank decomposition using randomized algebraic geometry approach.
        Sample random linear combinations of tensor slices to construct polynomial systems,
        solve for factor matrices over rationals using exact Gaussian elimination.
        """
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n

        # The tensor T is a 3-tensor of size nm x mp x pn
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for valid i,j,k, else 0
        # Note: each (i,j,k) maps to exactly one (a,b,c)

        # For rank r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
        # This means for each (a,b,c) in support: sum_{t=1}^r U[a,t]*V[b,t]*W[c,t] = 1
        # And for (a,b,c) not in support: sum_{t=1}^r U[a,t]*V[b,t]*W[c,t] = 0

        # Strategy: Fix U and V randomly (with rational entries), then solve for W.
        # Given U (nm x r) and V (mp x r), for each c in [0, pn):
        #   We need: sum_{t=1}^r W[c,t] * (U[:,t] ⊙ V[:,t])[c_index] = T[a,b,c] for all (a,b,c)
        # But this is not quite right. Let me reformulate.
        #
        # For a fixed c (output index), the constraint is:
        #   For all (i,j,k) with k*n+i = c: sum_t U[i*m+j, t] * V[j*p+k, t] * W[c, t] = 1
        #
        # So for each c, we have constraints on W[c, :].
        # Let's group by c = k*n + i.
        # For fixed k and i (so c = k*n+i), and for each j in [0,m):
        #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[c, t] = 1
        #
        # This gives m linear equations in W[c, 0], ..., W[c, r-1].
        # So for each c, we have a system A_c * W[c, :] = ones(m)
        # where A_c[j, t] = U[i*m+j, t] * V[j*p+k, t] for j in [0,m)

        # But we also need to ensure that for (a,b,c) NOT in the support, the sum is 0.
        # However, if we only enforce the support constraints, we might get spurious solutions.
        # Let's first try just the support constraints and then verify.

        for attempt in range(max_attempts):
            random.seed(attempt * 12345 + 67890)
            np.random.seed(attempt * 12345 + 67890)

            # Initialize U and V with small random rational entries
            # Use entries from {-2, -1, 0, 1, 2} to keep things simple
            U = np.zeros((nm, target_rank), dtype=object)
            V = np.zeros((mp, target_rank), dtype=object)

            for a in range(nm):
                for t in range(target_rank):
                    U[a, t] = random.randint(-2, 2)
            for b in range(mp):
                for t in range(target_rank):
                    V[b, t] = random.randint(-2, 2)

            # For each c in [0, pn), solve for W[c, :]
            W = np.zeros((pn, target_rank), dtype=object)
            feasible = True

            for c in range(pn):
                # c = k*n + i
                k = c // n
                i = c % n

                # Build system: for each j in [0,m): sum_t U[i*m+j, t] * V[j*p+k, t] * W[c, t] = 1
                A = np.zeros((m, target_rank), dtype=object)
                for j in range(m):
                    for t in range(target_rank):
                        A[j, t] = U[i * m + j, t] * V[j * p + k, t]

                # Solve A * W[c, :] = ones(m)
                # Use Gaussian elimination over rationals
                try:
                    W_c = _solve_linear_system(A, [1] * m)
                    if W_c is None:
                        feasible = False
                        break
                    for t in range(target_rank):
                        W[c, t] = W_c[t]
                except:
                    feasible = False
                    break

            if not feasible:
                continue

            # Convert to list of lists with integer/fraction entries
            U_list = [[U[a, t] for t in range(target_rank)] for a in range(nm)]
            V_list = [[V[b, t] for t in range(target_rank)] for b in range(mp)]
            W_list = [[W[c, t] for t in range(target_rank)] for c in range(pn)]

            # Verify the full decomposition
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                # Convert to proper types (int if integer, Fraction otherwise)
                def convert_entry(x):
                    if isinstance(x, Fraction):
                        if x.denominator == 1:
                            return int(x.numerator)
                        return x
                    elif isinstance(x, (int, np.integer)):
                        return int(x)
                    else:
                        return x
                U_out = [[convert_entry(x) for x in row] for row in U_list]
                V_out = [[convert_entry(x) for x in row] for row in V_list]
                W_out = [[convert_entry(x) for x in row] for row in W_list]
                return {"U": U_out, "V": V_out, "W": W_out}

        return None

    def _solve_linear_system(A, b):
        """
        Solve A * x = b over the rationals using Gaussian elimination.
        A is (m x r), b is length m. Returns solution x of length r, or None if no solution.
        """
        m = len(b)
        r = len(A[0]) if m > 0 else 0

        # Augmented matrix
        aug = [[A[i, j] for j in range(r)] + [b[i]] for i in range(m)]

        # Gaussian elimination
        row = 0
        for col in range(r):
            # Find pivot
            pivot = -1
            for i in range(row, m):
                if aug[i][col] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue

            # Swap rows
            aug[row], aug[pivot] = aug[pivot], aug[row]

            # Eliminate below
            for i in range(row + 1, m):
                if aug[i][col] != 0:
                    factor = aug[i][col] / aug[row][col]
                    for j in range(col, r + 1):
                        aug[i][j] = aug[i][j] - factor * aug[row][j]
            row += 1
            if row == m:
                break

        # Check for inconsistency
        for i in range(row, m):
            if aug[i][r] != 0:
                return None

        # Back substitution
        x = [0] * r
        for i in range(row - 1, -1, -1):
            x[i] = aug[i][r]
            for j in range(i + 1, r):
                x[i] = x[i] - aug[i][j] * x[j]
            if aug[i][i] != 0:
                x[i] = x[i] / aug[i][i]

        return x

    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # Lower bound: max(nm, mp, pn) = 21
    lower_bound = max(nm, mp, pn)

    # Try a few ranks below schoolbook
    for r in range(schoolbook - 1, lower_bound - 1, -1):
        result = try_rank_reduction(r, max_attempts=30)
        if result is not None:
            return result

    # Fall back to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END


"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Build a rank-44 decomposition using Strassen (7) for each 2x2 sub-block
     of the 2x2x8 part and schoolbook (16) for the 2x1x8 part.
  2. Use a greedy exact-integer elimination procedure to remove redundant
     terms: for each term r, test whether it can be expressed as a linear
     combination of the remaining terms with integer coefficients.
  3. Also try two-term simultaneous elimination to find pairs of terms that
     are jointly redundant but individually irreducible.
  4. Return the best (lowest-rank) valid decomposition found within the time budget.
"""
import time
import sys
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Validation ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        if not U or not V or not W:
            return False
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv != 0:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        # Check for any float entries that are not integers
        for mat in (U, V, W):
            for row in mat:
                for v in row:
                    if isinstance(v, float):
                        if v != int(v):
                            return False
        return True

    # ---- Build rank-44 Strassen + schoolbook decomposition ----
    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    # Verify Strassen internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank_init = 44
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]

    col = 0

    # Part 1: A[0:2,0:2] * B[0:2,0:8] via 4 Strassen blocks
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A[0:2,2] * B[2,0:8] via schoolbook
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # ---- Greedy rank reduction via exact integer linear algebra ----
    # We attempt to eliminate terms one at a time or two at a time.

    def _try_eliminate_term(U, V, W, term_to_remove):
        """
        Try to express term `term_to_remove` as an integer linear combination
        of the other terms. Returns (coefficients_dict, True) on success,
        (None, False) on failure.
        """
        rank = len(U[0])
        keep_terms = [j for j in range(rank) if j != term_to_remove]
        num_keep = len(keep_terms)

        # Build the linear system: for each (i,a,k), we have an equation
        # sum_j c_j * (U[i,a,j]*V[a,k,j]*W[k,i,j]) = U[i,a,term]*V[a,k,term]*W[k,i,term]
        # Number of equations: n*m*p = 48
        # Number of unknowns: num_keep

        # Collect equations
        equations = []  # each is (coeff_vector, rhs)
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    rhs = Fraction(U[i * m + a][term_to_remove] *
                                   V[a * p + k][term_to_remove] *
                                   W[k * n + i][term_to_remove])
                    coeff_vec = []
                    for j in keep_terms:
                        coeff_vec.append(Fraction(U[i * m + a][j] *
                                                  V[a * p + k][j] *
                                                  W[k * n + i][j]))
                    equations.append((coeff_vec, rhs))

        # Gaussian elimination over Q
        # Augmented matrix: num_equations x (num_keep + 1)
        num_eq = len(equations)
        # Use a list of lists for the augmented matrix
        aug = []
        for cv, rhs in equations:
            row = list(cv) + [rhs]
            aug.append(row)

        # Perform Gaussian elimination
        rank_aug = 0
        pivot_cols = []
        for col_idx in range(num_keep):
            # Find pivot
            pivot_row = -1
            for row_idx in range(rank_aug, num_eq):
                if aug[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            # Swap
            aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
            pivot_cols.append(col_idx)
            # Normalize
            pivot_val = aug[rank_aug][col_idx]
            for j in range(col_idx, num_keep + 1):
                aug[rank_aug][j] = aug[rank_aug][j] / pivot_val
            # Eliminate
            for row_idx in range(num_eq):
                if row_idx == rank_aug:
                    continue
                factor = aug[row_idx][col_idx]
                if factor != 0:
                    for j in range(col_idx, num_keep + 1):
                        aug[row_idx][j] -= factor * aug[rank_aug][j]
            rank_aug += 1
            if not _elapsed_ok():
                return None, False

        # Check consistency: any row with all-zero coefficients but non-zero rhs?
        for row_idx in range(rank_aug, num_eq):
            if aug[row_idx][num_keep] != 0:
                return None, False

        # Extract solution
        # The solution is determined by the pivot columns; free variables are 0.
        sol = [Fraction(0)] * num_keep
        for pcol, pidx in enumerate(pivot_cols):
            sol[pidx] = aug[pcol][num_keep]

        # Check if solution is integer-valued
        for val in sol:
            if val != int(val):
                return None, False

        # Map back to original term indices
        coeff_dict = {}
        for idx, j in enumerate(keep_terms):
            if sol[idx] != 0:
                coeff_dict[j] = int(sol[idx])

        return coeff_dict, True

    def _apply_elimination(U, V, W, term_to_remove, coeff_dict):
        """
        Remove term `term_to_remove` by adding coeff_dict[j] * term_j to term_to_remove's
        contribution. The decomposition is:
        sum_r U[r]*V[r]*W[r] = tensor
        If term_r = sum_{j!=r} c_j * term_j, then:
        sum_r = sum_{j!=r} term_j + term_r = sum_{j!=r} term_j + sum_{j!=r} c_j * term_j
              = sum_{j!=r} (1 + c_j) * term_j

        So we can drop term r and multiply each remaining term j by (1 + c_j).
        """
        rank = len(U[0])
        new_U = []
        new_V = []
        new_W = []
        for j in range(rank):
            if j == term_to_remove:
                continue
            scale = 1 + coeff_dict.get(j, 0)
            if scale == 0:
                # This term becomes zero; skip it
                continue
            new_U.append([u * scale for u in U[j]])
            new_V.append([v * scale for v in V[j]])
            new_W.append([w * scale for w in W[j]])
        return new_U, new_V, new_W

    def _try_eliminate_two_terms(U, V, W, term1, term2):
        """
        Try to express terms `term1` and `term2` together as an integer linear
        combination of the other terms. This allows finding pairs of terms that
        are jointly redundant but individually irreducible.
        """
        rank = len(U[0])
        keep_terms = [j for j in range(rank) if j != term1 and j != term2]
        num_keep = len(keep_terms)

        # Build the linear system: for each (i,a,k), we have an equation
        # sum_j c_j * (U[i,a,j]*V[a,k,j]*W[k,i,j]) = 
        #   U[i,a,term1]*V[a,k,term1]*W[k,i,term1] + U[i,a,term2]*V[a,k,term2]*W[k,i,term2]
        # Number of equations: n*m*p = 48
        # Number of unknowns: num_keep

        # Collect equations
        equations = []  # each is (coeff_vector, rhs)
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    rhs = Fraction(U[i * m + a][term1] *
                                   V[a * p + k][term1] *
                                   W[k * n + i][term1] +
                                   U[i * m + a][term2] *
                                   V[a * p + k][term2] *
                                   W[k * n + i][term2])
                    coeff_vec = []
                    for j in keep_terms:
                        coeff_vec.append(Fraction(U[i * m + a][j] *
                                                  V[a * p + k][j] *
                                                  W[k * n + i][j]))
                    equations.append((coeff_vec, rhs))

        # Gaussian elimination over Q
        # Augmented matrix: num_equations x (num_keep + 1)
        num_eq = len(equations)
        # Use a list of lists for the augmented matrix
        aug = []
        for cv, rhs in equations:
            row = list(cv) + [rhs]
            aug.append(row)

        # Perform Gaussian elimination
        rank_aug = 0
        pivot_cols = []
        for col_idx in range(num_keep):
            # Find pivot
            pivot_row = -1
            for row_idx in range(rank_aug, num_eq):
                if aug[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            # Swap
            aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
            pivot_cols.append(col_idx)
            # Normalize
            pivot_val = aug[rank_aug][col_idx]
            for j in range(col_idx, num_keep + 1):
                aug[rank_aug][j] = aug[rank_aug][j] / pivot_val
            # Eliminate
            for row_idx in range(num_eq):
                if row_idx == rank_aug:
                    continue
                factor = aug[row_idx][col_idx]
                if factor != 0:
                    for j in range(col_idx, num_keep + 1):
                        aug[row_idx][j] -= factor * aug[rank_aug][j]
            rank_aug += 1
            if not _elapsed_ok():
                return None, False

        # Check consistency: any row with all-zero coefficients but non-zero rhs?
        for row_idx in range(rank_aug, num_eq):
            if aug[row_idx][num_keep] != 0:
                return None, False

        # Extract solution
        # The solution is determined by the pivot columns; free variables are 0.
        sol = [Fraction(0)] * num_keep
        for pcol, pidx in enumerate(pivot_cols):
            sol[pidx] = aug[pcol][num_keep]

        # Check if solution is integer-valued
        for val in sol:
            if val != int(val):
                return None, False

        # Map back to original term indices
        coeff_dict = {}
        for idx, j in enumerate(keep_terms):
            if sol[idx] != 0:
                coeff_dict[j] = int(sol[idx])

        return coeff_dict, True

    def _apply_two_elimination(U, V, W, term1, term2, coeff_dict):
        """
        Remove terms `term1` and `term2` together. The decomposition is:
        sum_r U[r]*V[r]*W[r] = tensor
        If term1 + term2 = sum_{j!=term1,term2} c_j * term_j, then:
        sum_r = sum_{j!=term1,term2} term_j + term1 + term2
              = sum_{j!=term1,term2} term_j + sum_{j!=term1,term2} c_j * term_j
              = sum_{j!=term1,term2} (1 + c_j) * term_j

        So we can drop terms term1 and term2 and multiply each remaining term j by (1 + c_j).
        """
        rank = len(U[0])
        new_U = []
        new_V = []
        new_W = []
        for j in range(rank):
            if j == term1 or j == term2:
                continue
            scale = 1 + coeff_dict.get(j, 0)
            if scale == 0:
                # This term becomes zero; skip it
                continue
            new_U.append([u * scale for u in U[j]])
            new_V.append([v * scale for v in V[j]])
            new_W.append([w * scale for w in W[j]])
        return new_U, new_V, new_W

    # Greedy elimination loop
    best_U, best_V, best_W = U, V, W
    best_rank = rank_init

    # Phase 1: Single-term elimination
    progress = True
    while progress and _elapsed_ok():
        progress = False
        current_rank = len(best_U[0]) if best_U else 0
        for r in range(current_rank):
            if not _elapsed_ok():
                break
            coeff_dict, success = _try_eliminate_term(best_U, best_V, best_W, r)
            if success:
                new_U, new_V, new_W = _apply_elimination(best_U, best_V, best_W, r, coeff_dict)
                new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(new_candidate):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(new_U[0]) if new_U else 0
                    progress = True
                    if not _elapsed_ok():
                        break

    # Phase 2: Two-term elimination
    if _elapsed_ok() and best_rank > 40:
        progress = True
        while progress and _elapsed_ok():
            progress = False
            current_rank = len(best_U[0]) if best_U else 0
            for r1 in range(current_rank):
                for r2 in range(r1 + 1, current_rank):
                    if not _elapsed_ok():
                        break
                    coeff_dict, success = _try_eliminate_two_terms(best_U, best_V, best_W, r1, r2)
                    if success:
                        new_U, new_V, new_W = _apply_two_elimination(best_U, best_V, best_W, r1, r2, coeff_dict)
                        new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(new_candidate):
                            best_U, best_V, best_W = new_U, new_V, new_W
                            best_rank = len(new_U[0]) if new_U else 0
                            progress = True
                            if not _elapsed_ok():
                                break
                if not _elapsed_ok():
                    break

    # Phase 3: Final single-term pass after two-term eliminations
    if _elapsed_ok() and best_rank > 40:
        progress = True
        while progress and _elapsed_ok():
            progress = False
            current_rank = len(best_U[0]) if best_U else 0
            for r in range(current_rank):
                if not _elapsed_ok():
                    break
                coeff_dict, success = _try_eliminate_term(best_U, best_V, best_W, r)
                if success:
                    new_U, new_V, new_W = _apply_elimination(best_U, best_V, best_W, r, coeff_dict)
                    new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(new_candidate):
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = len(new_U[0]) if new_U else 0
                        progress = True
                        if not _elapsed_ok():
                            break

    if best_rank < rank_init:
        result = {"U": best_U, "V": best_V, "W": best_W}
        if _validate(result):
            return result

    return candidate

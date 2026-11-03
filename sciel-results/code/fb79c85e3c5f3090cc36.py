"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _rational_solve(A, b):
    """Solve linear system Ax = b over rationals using Gaussian elimination.
    Returns solution vector or None if no unique solution.
    A is (m, n) matrix of Fractions, b is (m,) vector of Fractions.
    """
    m, n = A.shape
    # Augmented matrix
    M = [list(A[i]) + [b[i]] for i in range(m)]
    
    # Gaussian elimination with partial pivoting
    for col in range(min(m, n)):
        # Find pivot
        pivot_row = None
        for row in range(col, m):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None  # No unique solution
        
        # Swap rows
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Eliminate below
        pivot_val = M[col][col]
        for row in range(col + 1, m):
            factor = M[row][col] / pivot_val
            for j in range(col, m + 1):
                M[row][j] -= factor * M[col][j]
    
    # Back substitution
    x = [Fraction(0)] * n
    for row in range(min(m, n) - 1, -1, -1):
        # Check if this row is all zeros in coefficient part
        all_zero = True
        for j in range(row, n):
            if M[row][j] != 0:
                all_zero = False
                break
        if all_zero:
            if M[row][n] != 0:
                return None  # Inconsistent
            continue
        
        s = M[row][n]
        for j in range(row + 1, n):
            s -= M[row][j] * x[j]
        x[row] = s / M[row][row]
    
    # Verify solution
    for i in range(m):
        s = Fraction(0)
        for j in range(n):
            s += A[i][j] * x[j]
        if s != b[i]:
            return None
    
    return x

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Standard Strassen 2x2x2 block
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B10 - B00)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B01 - B11)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_structured_linear_solve(n, m, p, max_rank=54):
    """
    Try to find a lower-rank decomposition by fixing U and V to sparse {0, ±1} 
    patterns and solving for W over rationals.
    
    Strategy: Start from the 54-rank Strassen decomposition, then try removing
    terms one at a time and check if the remaining system has a rational solution
    for W (with U and V fixed).
    
    Returns (U, V, W, rank) if successful, else None.
    """
    # Get the base 54-rank decomposition
    U_base, V_base, W_base = _build_54_rank(n, m, p)
    base_rank = U_base.shape[1]
    
    # Try ranks from base_rank down to max(1, base_rank - 10)
    for target_rank in range(base_rank, max(1, base_rank - 10), -1):
        if _time_left() < 20:
            break
        
        # Try different subsets of terms to keep
        # We'll try removing (base_rank - target_rank) terms
        n_remove = base_rank - target_rank
        
        if n_remove == 0:
            # Just verify the base decomposition
            if _verify_full(U_base, V_base, W_base, n, m, p):
                return U_base, V_base, W_base, base_rank
            continue
        
        # Generate candidate removal sets
        # Try all terms for n_remove=1, small-norm for larger
        if n_remove == 1:
            candidates = [tuple([t]) for t in range(base_rank)]
        else:
            norms = []
            for r in range(base_rank):
                norm_sq = np.sum(U_base[:, r]**2) + np.sum(V_base[:, r]**2) + np.sum(W_base[:, r]**2)
                norms.append(norm_sq)
            small_terms = np.argsort(norms)[:min(15, base_rank)]
            if n_remove == 2:
                candidates = [tuple(t) for t in itertools.combinations(small_terms, 2)]
            elif n_remove == 3:
                candidates = [tuple(t) for t in itertools.combinations(small_terms[:10], 3)]
            else:
                continue  # Too many combinations for larger n_remove
        
        for remove_set in candidates:
            if _time_left() < 10:
                break
            
            # Create mask for kept terms
            keep_mask = np.ones(base_rank, dtype=bool)
            for r in remove_set:
                keep_mask[r] = False
            
            U_test = U_base[:, keep_mask]
            V_test = V_base[:, keep_mask]
            
            # Build the linear system for W
            n_unknowns = p * n * target_rank
            
            # Use all valid equations plus a subset of invalid equations
            n_valid_eqs = n * m * p
            # For invalid equations, we need enough to constrain the solution
            # But too many makes the solve slow. Use a balanced approach.
            n_invalid_eqs = min(200, (p*n) * (n*m) * (m*p) - n*m*p)
            
            total_eqs = n_valid_eqs + n_invalid_eqs
            
            # Build A and b using Fractions for exact arithmetic
            A_list = []
            b_list = []
            
            # Valid equations
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = [Fraction(0)] * n_unknowns
                        w_base = (k * n + i) * target_rank
                        for r in range(target_rank):
                            coeff = U_test[i*m+j, r] * V_test[j*p+k, r]
                            row[w_base + r] = Fraction(int(coeff))
                        A_list.append(row)
                        b_list.append(Fraction(1))
            
            # Invalid equations (sample some)
            if n_invalid_eqs > 0:
                # Generate some invalid (x,y,z) triples
                invalid_triples = []
                for x in range(p*n):
                    i = x % n
                    k = x // n
                    for y in range(n*m):
                        j = y % m
                        i2 = y // m
                        if i == i2:
                            for z in range(m*p):
                                j2 = z // p
                                k2 = z % p
                                if j == j2 and k == k2:
                                    continue
                                invalid_triples.append((x, y, z))
                
                if invalid_triples:
                    step = max(1, len(invalid_triples) // n_invalid_eqs)
                    for idx in range(0, len(invalid_triples), step):
                        if len(A_list) >= total_eqs:
                            break
                        x, y, z = invalid_triples[idx]
                        row = [Fraction(0)] * n_unknowns
                        w_base = x * target_rank
                        for r in range(target_rank):
                            coeff = U_test[y, r] * V_test[z, r]
                            row[w_base + r] = Fraction(int(coeff))
                        A_list.append(row)
                        b_list.append(Fraction(0))
            
            # Solve the system
            A_mat = np.array(A_list, dtype=object)
            b_vec = np.array(b_list, dtype=object)
            
            solution = _rational_solve(A_mat, b_vec)
            
            if solution is not None:
                # Reconstruct W
                W_test = np.zeros((p*n, target_rank))
                for x in range(p*n):
                    for r in range(target_rank):
                        W_test[x, r] = float(solution[x * target_rank + r])
                
                # Verify the solution
                if _verify_full(U_test, V_test, W_test, n, m, p, tol=1e-6):
                    U_clean = U_test.copy()
                    V_clean = V_test.copy()
                    W_clean = W_test.copy()
                    return U_clean, V_clean, W_clean, target_rank
            
            if _time_left() < 5:
                break
    
    return None

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using vectorized numpy operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each valid (i,j,k) triple, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries: for all (x,y,z) where the indices don't form a valid triple
    # x = k*n+i, y = i2*m+j, z = j2*p+k2
    # Valid if i==i2, j==j2, k==k2
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                if abs(s) > tol:
                    return False
    return True

def _verify_skip_term(U, V, W, n, m, p, skip_term, tol=1e-9):
    """
    Verify that the decomposition remains valid if term skip_term is set to zero.
    """
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s_full = np.dot(u_row, v_row * w_row)
                s_skip = u_row[skip_term] * v_row[skip_term] * w_row[skip_term]
                if abs(s_full - s_skip - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s_full = np.dot(U[y, :], V[z, :] * W[x, :])
                s_skip = U[y, skip_term] * V[z, skip_term] * W[x, skip_term]
                if abs(s_full - s_skip) > tol:
                    return False
    return True

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one.
    Returns (U, V, W, new_rank).
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    eliminated = []
    
    # Try eliminating each term
    # Try all terms, not just small-norm ones, to be more thorough
    for r in range(current_rank):
        if _time_left() < 30:
            break
        if len(eliminated) >= max_eliminations:
            break
        
        # Check if we can eliminate term r
        if _verify_skip_term(current_U, current_V, current_W, n, m, p, r):
            # Eliminate this term
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            eliminated.append(r)
            # After removing a term, re-check from the beginning since indices shift
            # But for efficiency, continue with the remaining terms
            # We'll just continue the loop; the next r will be the next original index
            # which is now shifted, but that's fine since we're checking all terms
    
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously. This can sometimes reduce rank
    further than single term elimination.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Try all pairs if time permits
    # Number of pairs is R*(R-1)/2. For R=54, that's 1431 pairs.
    # Each check is O(nmp + pnm * nm * mp) which is fast.
    
    # Try all pairs, not just small-norm ones
    for r1 in range(current_rank):
        if _time_left() < 20:
            break
        for r2 in range(r1 + 1, current_rank):
            # Create masks excluding r1 and r2
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_full(U_test, V_test, W_test, n, m, p):
                # Success! Remove both
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 2
                return current_U, current_V, current_W, current_rank
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by finding linear dependencies among the terms.
    This is a heuristic approach using the tensor structure.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    removed = []
    
    for r1 in range(current_rank):
        if r1 in removed:
            continue
        if _time_left() < 20:
            break
        for r2 in range(r1+1, current_rank):
            if r2 in removed:
                continue
            
            # Check if term r2 is proportional to term r1 in all three matrices
            u1 = current_U[:, r1]
            u2 = current_U[:, r2]
            v1 = current_V[:, r1]
            v2 = current_V[:, r2]
            w1 = current_W[:, r1]
            w2 = current_W[:, r2]
            
            if np.allclose(u1, 0) and np.allclose(u2, 0):
                continue
            
            mask = np.abs(u2) > 1e-10
            if not np.any(mask):
                continue
            
            alpha = u1[mask] / u2[mask]
            if not np.allclose(alpha, alpha[0], atol=1e-8):
                continue
            alpha = alpha[0]
            
            if np.allclose(u1, alpha * u2, atol=1e-8) and \
               np.allclose(v1, alpha * v2, atol=1e-8) and \
               np.allclose(w1, alpha * w2, atol=1e-8):
                # Term r1 is proportional to term r2
                # We can't simply remove one; we need to check if the combination
                # with other terms still works. For now, skip this complex case.
                pass
    
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try to eliminate redundant terms (single) - do this first as it's faster
    if _time_left() > 60:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try to eliminate pairs of terms
    if _time_left() > 30:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try structured linear algebra approach (fix U,V, solve for W) - slower, do last
    if _time_left() > 50:
        result = _try_structured_linear_solve(n, m, p)
        if result is not None:
            U_sol, V_sol, W_sol, new_rank = result
            if new_rank < best_rank:
                if _verify_full(U_sol, V_sol, W_sol, n, m, p):
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = new_rank
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

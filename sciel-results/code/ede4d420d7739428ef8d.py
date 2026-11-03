"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import combinations, product

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _snap_to_half_integer(x, tol=1e-8):
    """Snap a value to the nearest half-integer if close enough."""
    h = round(2 * x) / 2.0
    if abs(x - h) < tol:
        return h
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

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using vectorized numpy operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    
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

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=10):
    """
    Try to eliminate redundant terms one by one.
    Returns (U, V, W, new_rank).
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    eliminated = []
    
    # Try eliminating each term, starting with those that might be redundant
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
    
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 30:
            break
        if len(eliminated) >= max_eliminations:
            break
        
        if _verify_skip_term(current_U, current_V, current_W, n, m, p, r):
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            eliminated.append(r)
    
    return current_U, current_V, current_W, current_rank

def _build_tensor_slices(U, V, W, n, m, p):
    """
    Build the tensor T with T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    Returns a 3D numpy array of shape (n*m, m*p, p*n)
    """
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        T += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    return T

def _try_mpo_reduction(U, V, W, n, m, p):
    """
    Use SVD-based MPO gauge fixing to identify and eliminate redundant terms.
    This works by viewing the tensor as a matrix product operator and using
    singular value decomposition to find the minimal bond dimension.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Build the full tensor
    T = _build_tensor_slices(current_U, current_V, current_W, n, m, p)
    
    # The tensor T has shape (n*m, m*p, p*n) = (10, 30, 12)
    # We want to find a low-rank decomposition
    
    # Strategy: Try to find a rank-R decomposition where R < current_rank
    # by using iterative SVD-based compression
    
    # First, let's try to see if we can reduce by checking the "effective rank"
    # of the tensor when viewed as a matrix in different ways
    
    # View T as a matrix (n*m) x (m*p * p*n)
    T_mat1 = T.reshape(n*m, -1)
    # View T as a matrix (m*p) x (n*m * p*n)
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, -1)
    # View T as a matrix (p*n) x (n*m * m*p)
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, -1)
    
    # Compute singular values to estimate the effective rank
    sv1 = np.linalg.svd(T_mat1, compute_uv=False)
    sv2 = np.linalg.svd(T_mat2, compute_uv=False)
    sv3 = np.linalg.svd(T_mat3, compute_uv=False)
    
    # The tensor rank is at least the maximum of these matrix ranks
    # But we need to be more careful - the actual tensor rank could be higher
    
    # Try a greedy approach: iteratively remove terms that contribute least
    # to the tensor energy, while checking validity
    
    # Sort terms by their contribution to the tensor norm
    term_norms = []
    for r in range(current_rank):
        u_col = current_U[:, r]
        v_col = current_V[:, r]
        w_col = current_W[:, r]
        # Norm of the rank-1 term
        norm = np.linalg.norm(u_col) * np.linalg.norm(v_col) * np.linalg.norm(w_col)
        term_norms.append(norm)
    
    term_order = np.argsort(term_norms)
    
    # Try removing terms in order of increasing norm
    removed = []
    for r in term_order:
        if _time_left() < 20:
            break
        
        # Skip if already removed
        if r in removed:
            continue
        
        # Check if we can remove term r
        if _verify_skip_term(current_U, current_V, current_W, n, m, p, r):
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            removed.append(r)
    
    return current_U, current_V, current_W, current_rank

def _try_pairwise_combination(U, V, W, n, m, p):
    """
    Try to combine pairs of terms into a single term using linear algebra.
    If two terms r1 and r2 can be combined, we reduce the rank by 1.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # For each pair of terms, check if their difference is a valid rank-1 term
    # that can replace them
    removed = set()
    
    for r1 in range(current_rank):
        if r1 in removed:
            continue
        if _time_left() < 15:
            break
        
        for r2 in range(r1+1, current_rank):
            if r2 in removed:
                continue
            
            # Check if terms r1 and r2 can be merged
            # This requires finding a new term r_new such that:
            # U[:, r1] + U[:, r2] = U[:, r_new] (or some linear combination)
            # and similarly for V and W
            # This is generally not possible, but let's check for special cases
            
            # Check if the two terms are "orthogonal" in a way that allows merging
            u1 = current_U[:, r1]
            u2 = current_U[:, r2]
            v1 = current_V[:, r1]
            v2 = current_V[:, r2]
            w1 = current_W[:, r1]
            w2 = current_W[:, r2]
            
            # If one term is a scalar multiple of the other, we can merge them
            # Check proportionality
            if np.linalg.norm(u1) < 1e-10 or np.linalg.norm(u2) < 1e-10:
                continue
            
            # Find scalar alpha such that u1 ≈ alpha * u2
            alpha = np.dot(u1, u2) / np.dot(u2, u2)
            if np.linalg.norm(u1 - alpha * u2) < 1e-8:
                # Check V and W
                if np.linalg.norm(v1 - alpha * v2) < 1e-8 and \
                   np.linalg.norm(w1 - alpha * w2) < 1e-8:
                    # Terms are proportional, merge them
                    current_U[:, r1] = u1 + u2
                    current_V[:, r1] = v1 + v2
                    current_W[:, r1] = w1 + w2
                    # Remove r2
                    mask = np.ones(current_rank, dtype=bool)
                    mask[r2] = False
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_rank -= 1
                    removed.add(r2)
                    break
    
    return current_U, current_V, current_W, current_rank

def _snap_all_to_half_integers(U, V, W, tol=1e-8):
    """Snap all entries to half-integers if close enough."""
    U_snap = np.array([[_snap_to_half_integer(x, tol) for x in row] for row in U])
    V_snap = np.array([[_snap_to_half_integer(x, tol) for x in row] for row in V])
    W_snap = np.array([[_snap_to_half_integer(x, tol) for x in row] for row in W])
    return U_snap, V_snap, W_snap

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
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
    
    # Try to eliminate redundant terms
    if _time_left() > 50:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
    
    # Try MPO-based reduction
    if _time_left() > 40:
        U_mpo, V_mpo, W_mpo, new_rank = _try_mpo_reduction(U, V, W, n, m, p)
        if new_rank < best_rank:
            if _verify_full(U_mpo, V_mpo, W_mpo, n, m, p):
                best_U, best_V, best_W = U_mpo, V_mpo, W_mpo
                best_rank = new_rank
    
    # Try pairwise combination
    if _time_left() > 30:
        U_pair, V_pair, W_pair, new_rank = _try_pairwise_combination(U, V, W, n, m, p)
        if new_rank < best_rank:
            if _verify_full(U_pair, V_pair, W_pair, n, m, p):
                best_U, best_V, best_W = U_pair, V_pair, W_pair
                best_rank = new_rank
    
    # Snap to half-integers
    best_U, best_V, best_W = _snap_all_to_half_integers(best_U, best_V, best_W)
    
    # Final verification
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        # Revert to original if snapping broke it
        best_U, best_V, best_W = U, V, W
        best_rank = 54
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

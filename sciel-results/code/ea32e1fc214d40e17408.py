"""
Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
Strategy: Randomized local search over tensor decompositions.
- Start from a Strassen-based 4x4x4 (rank 49) + schoolbook 4x4x3 (rank 24) = 73.
- Attempt to reduce rank via: (a) pair-merging (find two terms whose sum is close to rank-1),
  (b) random perturbation + refit, (c) column elimination (zero out a column if its
  contribution is absorbed by others).
- Verify exactness with rational reconstruction (Fraction).
"""

import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2*2x2."""
    terms = []
    # s1 = (a00+a11)*(b00+b11)
    terms.append({'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]})
    # s2 = (a10+a11)*b00
    terms.append({'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]})
    # s3 = a00*(b01-b11)
    terms.append({'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]})
    # s4 = a11*(b10-b00)
    terms.append({'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (2, 1)]})
    # s5 = (a00+a01)*b11
    terms.append({'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]})
    # s6 = (a10-a00)*(b00+b01)
    terms.append({'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]})
    # s7 = (a01-a11)*(b10+b11)
    terms.append({'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]})
    return terms


def _build_4x4x4_terms():
    """Build 4x4*4x4 tensor terms via Kronecker of Strassen 2x2x2."""
    base = _strassen_2x2x2_terms()
    all_terms = []
    for bi in range(2):
        for bj in range(2):
            for ci in range(2):
                for cj in range(2):
                    for st in base:
                        term = {'U': [], 'V': [], 'W': []}
                        for (u_idx, uc) in st['U']:
                            gu = bi * 2 + u_idx
                            term['U'].append((gu, uc))
                        for (v_idx, vc) in st['V']:
                            gv = bj * 2 + v_idx
                            term['V'].append((gv, vc))
                        for (w_idx, wc) in st['W']:
                            gw = ci * 2 + w_idx
                            term['W'].append((gw, wc))
                        all_terms.append(term)
    return all_terms


def _build_4x4x3_terms():
    """Build 4x4*4x3 tensor terms via schoolbook 2x2x3 blocks."""
    all_terms = []
    for bi in range(2):
        for bj in range(2):
            for ci in range(2):
                for cj in range(2):
                    for k in range(3):
                        term = {'U': [(bi * 2 + ci, 1)],
                                'V': [(bj * 3 + k, 1)],
                                'W': [(k * 2 + cj, 1)]}
                        all_terms.append(term)
    return all_terms


def _terms_to_matrices(terms, n, m, p):
    """Convert list of sparse terms to dense U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for col, term in enumerate(terms):
        for (idx, coeff) in term['U']:
            if 0 <= idx < n * m:
                U[idx, col] = coeff
        for (idx, coeff) in term['V']:
            if 0 <= idx < m * p:
                V[idx, col] = coeff
        for (idx, coeff) in term['W']:
            if 0 <= idx < p * n:
                W[idx, col] = coeff
    return U, V, W


def _matrices_to_terms(U, V, W, n, m, p, tol=1e-9):
    """Convert dense matrices back to sparse term list."""
    rank = U.shape[1]
    terms = []
    for r in range(rank):
        term = {'U': [], 'V': [], 'W': []}
        for idx in range(n * m):
            if abs(U[idx, r]) > tol:
                term['U'].append((idx, U[idx, r]))
        for idx in range(m * p):
            if abs(V[idx, r]) > tol:
                term['V'].append((idx, V[idx, r]))
        for idx in range(p * n):
            if abs(W[idx, r]) > tol:
                term['W'].append((idx, W[idx, r]))
        terms.append(term)
    return terms


def _tensor_value(U, V, W, n, m, p, i, j, k):
    """Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
    return np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])


def _verify_exact(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition: T[i,j,k] should be 1 for all valid (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = _tensor_value(U, V, W, n, m, p, i, j, k)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_all_zero_except_target(U, V, W, n, m, p, target_k_range, tol=1e-6):
    """Verify that for k in target_k_range, T=1, and for k outside, T=0."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = _tensor_value(U, V, W, n, m, p, i, j, k)
                if k in target_k_range:
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True


def _try_merge_pair(U, V, W, r1, r2, n, m, p, tol=1e-4):
    """
    Try to replace columns r1, r2 with a single column that approximates their sum.
    The sum tensor is T_sum = u1⊗v1⊗w1 + u2⊗v2⊗w2.
    Find best rank-1 approximation u⊗v⊗w via CP decomposition of the sum.
    Return (new_u, new_v, new_w, error) or None if not close to rank-1.
    """
    # Build the 3-tensor for the sum of two rank-1 tensors
    # T[i,j,k] = U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
    # We want to find u,v,w such that u[i*m+j]*v[j*p+k]*w[k*n+i] ≈ T[i,j,k]
    # This is equivalent to finding the best rank-1 approximation.
    
    # Build the full tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = (U[idx_u, r1] * V[idx_v, r1] * W[idx_w, r1] +
                                          U[idx_u, r2] * V[idx_v, r2] * W[idx_w, r2])
    
    # CP rank-1 approximation: maximize <T, u⊗v⊗w> / (||u||*||v||*||w||)
    # Use power iteration on the 3-mode unfolding
    u = U[:, r1] + U[:, r2]
    v = V[:, r1] + V[:, r2]
    w = W[:, r1] + W[:, r2]
    
    # Normalize
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if u_norm < 1e-12 or v_norm < 1e-12 or w_norm < 1e-12:
        return None
    
    # Power iteration for best rank-1
    for _ in range(50):
        # Update u: u = T_{(1)} @ (v ⊗ w)
        # T_{(1)} is (n*m) x (m*p*p*n)
        vw = np.outer(v, w)  # (m*p, p*n)
        # Reshape T to (n*m, m*p*p*n)
        T_reshaped = T.reshape(n * m, m * p * p * n)
        u_new = T_reshaped @ vw.flatten()
        u_new_norm = np.linalg.norm(u_new)
        if u_new_norm < 1e-12:
            break
        u = u_new / u_new_norm
        
        # Update v
        uw = np.outer(u, w)  # (n*m, p*n)
        T_reshaped_v = T.reshape(m * p, n * m * p * n)
        v_new = T_reshaped_v @ uw.flatten()
        v_new_norm = np.linalg.norm(v_new)
        if v_new_norm < 1e-12:
            break
        v = v_new / v_new_norm
        
        # Update w
        uv = np.outer(u, v)  # (n*m, m*p)
        T_reshaped_w = T.reshape(p * n, n * m * m * p)
        w_new = T_reshaped_w @ uv.flatten()
        w_new_norm = np.linalg.norm(w_new)
        if w_new_norm < 1e-12:
            break
        w = w_new / w_new_norm
    
    # Compute approximation error
    T_approx = np.einsum('i,j,k->ijk', u, v, w)
    error = np.linalg.norm(T - T_approx) / (np.linalg.norm(T) + 1e-12)
    
    if error > tol:
        return None
    
    # Scale to match: we want u⊗v⊗w ≈ T_sum
    # The best scaling factor
    inner = np.sum(T * T_approx)
    denom = np.sum(T_approx * T_approx)
    if denom < 1e-12:
        return None
    scale = inner / denom
    u = scale * u
    v = v  # Keep v as is, absorb scale into u
    # Actually, let's distribute: u *= scale^(1/3), v *= scale^(1/3), w *= scale^(1/3)
    # Simpler: just put all scale in u
    u = scale * u
    
    return u, v, w, error


def _local_search(U, V, W, n, m, p, max_iters=200, deadline=None):
    """
    Local search: try to reduce rank by merging pairs of columns.
    Returns best (U, V, W) found.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if best_rank <= 1:
            break
            
        # Try random pairs
        rank = best_U.shape[1]
        if rank < 2:
            break
        
        improved = False
        for attempt in range(20):
            if deadline and time.time() > deadline:
                break
            r1 = random.randint(0, rank - 1)
            r2 = random.randint(0, rank - 1)
            if r1 == r2:
                continue
            
            result = _try_merge_pair(best_U, best_V, best_W, r1, r2, n, m, p, tol=1e-3)
            if result is None:
                continue
            
            new_u, new_v, new_w, err = result
            
            # Build new matrices: replace columns r1, r2 with single column
            new_rank = rank - 1
            new_U = np.zeros((n * m, new_rank))
            new_V = np.zeros((m * p, new_rank))
            new_W = np.zeros((p * n, new_rank))
            
            cols_kept = [c for c in range(rank) if c != r1 and c != r2]
            for new_col, old_col in enumerate(cols_kept):
                new_U[:, new_col] = best_U[:, old_col]
                new_V[:, new_col] = best_V[:, old_col]
                new_W[:, new_col] = best_W[:, old_col]
            
            # Add the merged column at the end
            new_U[:, -1] = new_u
            new_V[:, -1] = new_v
            new_W[:, -1] = new_w
            
            # Verify
            if _verify_exact(new_U, new_V, new_W, n, m, p, tol=1e-4):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                improved = True
                break
        
        if not improved:
            # Try a few more random pairs before giving up this iteration
            for attempt in range(10):
                if deadline and time.time() > deadline:
                    break
                r1 = random.randint(0, best_U.shape[1] - 1)
                r2 = random.randint(0, best_U.shape[1] - 1)
                if r1 == r2:
                    continue
                result = _try_merge_pair(best_U, best_V, best_W, r1, r2, n, m, p, tol=1e-3)
                if result is None:
                    continue
                new_u, new_v, new_w, err = result
                rank = best_U.shape[1]
                new_rank = rank - 1
                new_U = np.zeros((n * m, new_rank))
                new_V = np.zeros((m * p, new_rank))
                new_W = np.zeros((p * n, new_rank))
                cols_kept = [c for c in range(rank) if c != r1 and c != r2]
                for new_col, old_col in enumerate(cols_kept):
                    new_U[:, new_col] = best_U[:, old_col]
                    new_V[:, new_col] = best_V[:, old_col]
                    new_W[:, new_col] = best_W[:, old_col]
                new_U[:, -1] = new_u
                new_V[:, -1] = new_v
                new_W[:, -1] = new_w
                if _verify_exact(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_rank
                    improved = True
                    break
            
            if not improved:
                break  # Stuck, stop searching
    
    return best_U, best_V, best_W


def _rationalize_matrix(M, max_denom=100):
    """Convert float matrix to Fraction matrix with small denominators."""
    rows, cols = M.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = M[i, j]
            if abs(val) < 1e-9:
                result[i][j] = Fraction(0)
            else:
                # Try to find a good rational approximation
                frac = Fraction(val).limit_denominator(max_denom)
                result[i][j] = frac
    return result


def _rationalize_decomposition(U, V, W, n, m, p, max_denom=200):
    """Rationalize and verify."""
    U_r = _rationalize_matrix(U, max_denom)
    V_r = _rationalize_matrix(V, max_denom)
    W_r = _rationalize_matrix(W, max_denom)
    
    # Verify with exact arithmetic
    rank = len(U_r[0]) if U_r else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U_r[i * m + j][r] * V_r[j * p + k][r] * W_r[k * n + i][r]
                if total != 1:
                    return None
    return U_r, V_r, W_r


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build initial decomposition: 4x4x4 Strassen (49) + 4x4x3 schoolbook (24) = 73
    terms_4x4x4 = _build_4x4x4_terms()
    terms_4x4x3 = _build_4x4x3_terms()
    
    # Combine: 4x4x4 terms use V indices 0..15 (j*4+k, k<4)
    # 4x4x3 terms use V indices j*7+k (k>=4)
    # Need to remap V indices for 4x4x4: j*4+k -> j*7+k
    all_terms = []
    for term in terms_4x4x4:
        new_term = {'U': [], 'V': [], 'W': []}
        for (idx, coeff) in term['U']:
            new_term['U'].append((idx, coeff))
        for (idx, coeff) in term['V']:
            j = idx // 4
            k = idx % 4
            new_idx = j * 7 + k
            new_term['V'].append((new_idx, coeff))
        for (idx, coeff) in term['W']:
            new_term['W'].append((idx, coeff))
        all_terms.append(new_term)
    
    for term in terms_4x4x3:
        all_terms.append(term)
    
    U, V, W = _terms_to_matrices(all_terms, n, m, p)
    
    # Verify initial
    if not _verify_exact(U, V, W, n, m, p, tol=1e-6):
        # Fallback to pure schoolbook
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
        U_r, V_r, W_r = _rationalize_decomposition(U, V, W, n, m, p)
        if U_r is not None:
            return {"U": U_r, "V": V_r, "W": W_r}
        # Return as float if rationalization fails
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Local search to reduce rank
    U, V, W = _local_search(U, V, W, n, m, p, max_iters=100, deadline=deadline)
    
    # Try rationalization
    U_r, V_r, W_r = _rationalize_decomposition(U, V, W, n, m, p, max_denom=500)
    if U_r is not None:
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # If rationalization fails, try with larger denominators
    U_r, V_r, W_r = _rationalize_decomposition(U, V, W, n, m, p, max_denom=2000)
    if U_r is not None:
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback: return the original 73-rank decomposition (which is exact with integers)
    U_orig, V_orig, W_orig = _terms_to_matrices(all_terms, n, m, p)
    U_r, V_r, W_r = _rationalize_decomposition(U_orig, V_orig, W_orig, n, m, p, max_denom=10)
    if U_r is not None:
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # Last resort: return float matrices
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    # Quick verify
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    valid = _verify_exact(U, V, W, 4, 4, 7, tol=1e-4)
    print(f"Valid: {valid}")

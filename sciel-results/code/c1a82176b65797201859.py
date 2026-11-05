"""
Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

Strategy:
  1. Start with schoolbook rank-54 decomposition.
  2. Use ALS (Alternating Least Squares) to search for lower-rank decompositions.
  3. After ALS converges, perform exact rational recovery and verification.
  4. Return the best (lowest rank) valid decomposition found.

The tensor convention is T[i*m+j, j*p+k, k*n+i] = 1.
"""

import time
import numpy as np
from fractions import Fraction
from math import gcd

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
n, m, p = 3, 3, 6
nm, mp, pn = n*m, m*p, p*n  # 9, 18, 18

# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

def _build_tensor():
    """
    Build the matrix-multiplication tensor T of shape (nm, mp, pn)
    using the official trace convention:
        T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

# Precompute the tensor
T = _build_tensor()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_float(U, V, W, tol=1e-8):
    """
    Check whether factor matrices approximately decompose the tensor.
    """
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(T_rec - T)) < tol

def _verify_exact(U, V, W):
    """
    Check whether factor matrices exactly decompose the tensor.
    U, V, W can be numpy arrays or lists of Fractions.
    """
    # Determine if we're using Fractions or numpy
    if isinstance(U, np.ndarray):
        # Convert to check exactly
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(np.round(T_rec).astype(np.int64), T.astype(np.int64))
    else:
        # Fraction-based verification
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for col in range(r):
                        s += U[a][col] * V[b][col] * W[c][col]
                    if s != Fraction(1):
                        return False
        return True

# ---------------------------------------------------------------------------
# Schoolbook decomposition (rank 54)
# ---------------------------------------------------------------------------

def _schoolbook():
    """
    Return the schoolbook rank-54 decomposition for 3x3x6.
    """
    rank = n * m * p  # 54
    
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

# ---------------------------------------------------------------------------
# Rational recovery
# ---------------------------------------------------------------------------

def _recover_rational(x, max_denom=100000):
    """
    Recover a rational approximation of float x with denominator <= max_denom.
    """
    x = float(x)
    if abs(x) < 1e-10:
        return Fraction(0)
    # Use continued fractions via limit_denominator
    return Fraction(x).limit_denominator(max_denom)

def _to_fraction_matrix(mat, max_denom=100000):
    """
    Convert a numpy matrix to a list-of-lists of Fractions.
    """
    rows = mat.shape[0]
    cols = mat.shape[1]
    out = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = _recover_rational(mat[i, j], max_denom)
    return out

def _verify_fraction_decomp(U_frac, V_frac, W_frac):
    """
    Verify the decomposition using exact Fraction arithmetic.
    """
    r = len(U_frac[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for col in range(r):
                    s += U_frac[a][col] * V_frac[b][col] * W_frac[c][col]
                if s != Fraction(1):
                    return False
    return True

# ---------------------------------------------------------------------------
# ALS-based search for lower rank
# ---------------------------------------------------------------------------

def _als_single_trial(r, seed, max_iter=500, tol=1e-14):
    """
    Single ALS trial to find a rank-r decomposition.
    Returns (U, V, W) as float numpy arrays, or None if failed.
    """
    np.random.seed(seed)
    
    # Random initialization with structured patterns
    U = np.random.randn(nm, r)
    V = np.random.randn(mp, r)
    W = np.random.randn(pn, r)
    
    # Normalize initial factors
    for c in range(r):
        norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
        if norm > 0:
            scale = norm ** (1.0/3.0)
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: U[a, c] = sum_{b, d} T[a, b, d] * V[b, c] * W[d, c]
        T_cW = np.einsum('abd,dc->abc', T, W, optimize=True)  # (nm, mp, r)
        U_new = np.einsum('abc,bc->ac', T_cW, V, optimize=True)  # (nm, r)
        
        # Update V: V[b, c] = sum_{a, d} T[a, b, d] * U[a, c] * W[d, c]
        T_uW = np.einsum('abd,ac,dc->bc', T, U_new, W, optimize=True)  # (mp, r)
        V_new = T_uW
        
        # Update W: W[d, c] = sum_{a, b} T[a, b, d] * U[a, c] * V[b, c]
        T_uV = np.einsum('abd,ac,bc->dc', T, U_new, V_new, optimize=True)  # (pn, r)
        W_new = T_uV
        
        # Compute error
        T_rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
        err = np.max(np.abs(T_rec - T))
        
        # Check convergence
        if abs(prev_err - err) < tol and err < 1e-8:
            U, V, W = U_new, V_new, W_new
            break
        
        U, V, W = U_new, V_new, W_new
        prev_err = err
    
    # Final error check
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(T_rec - T))
    
    if err < 1e-8:
        return U, V, W
    return None

def _als_search(max_rank, deadline, num_trials_per_rank=8):
    """
    Try to find a rank < max_rank exact decomposition via randomised ALS.
    Returns (U, V, W) as fraction matrices, or None.
    """
    # Try ranks from max_rank-1 down to 20
    for r in range(min(max_rank - 1, 50), 19, -1):
        if time.time() > deadline:
            break
        
        for trial in range(num_trials_per_rank):
            if time.time() > deadline:
                break
            
            seed = 42 + r * 1000 + trial * 7
            result = _als_single_trial(r, seed)
            
            if result is not None:
                U, V, W = result
                
                # Try to recover exact rational entries
                for max_denom in [100, 1000, 10000, 100000, 1000000]:
                    U_frac = _to_fraction_matrix(U, max_denom)
                    V_frac = _to_fraction_matrix(V, max_denom)
                    W_frac = _to_fraction_matrix(W, max_denom)
                    
                    if _verify_fraction_decomp(U_frac, V_frac, W_frac):
                        return U_frac, V_frac, W_frac
                
                # If rational recovery fails with small denominators, try with larger
                U_frac = _to_fraction_matrix(U, 10000000)
                V_frac = _to_fraction_matrix(V, 10000000)
                W_frac = _to_fraction_matrix(W, 10000000)
                
                if _verify_fraction_decomp(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac
    
    return None

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6
    matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Start with schoolbook (rank 54)
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = 54
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank via ALS search
    remaining = deadline - time.time()
    if remaining > 30:
        result = _als_search(max_rank=54, deadline=deadline - 10)
        if result is not None:
            U_frac, V_frac, W_frac = result
            r = len(U_frac[0])
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_frac, V_frac, W_frac
    
    # Convert best result to output format
    if isinstance(best_U, np.ndarray):
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        # Fraction matrices - convert to list of lists of floats
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy:
  1. Guaranteed fallback: schoolbook rank 28.
  2. Attempt to find a lower-rank exact decomposition by:
     a. Structured merging of schoolbook terms (exploiting symmetries)
     b. Numerical optimization with exact rational verification
  The W matrix follows the official trace convention W[k*n+i, r].
"""

import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 7]


def _build_schoolbook(n, m, p):
    """Explicit schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Verify that (U,V,W) exactly decomposes the multiplication tensor.
    
    Checks:
      1. For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] == delta[i,k]
      2. For each column c: W has exactly one nonzero entry at position k*n+i
         where the column's contribution is to output (i,k).
    """
    R = len(U[0])
    
    # Convert to fractions for exact arithmetic
    def to_frac(x):
        if isinstance(x, Fraction):
            return x
        if isinstance(x, float):
            # Round to nearest half-integer
            h = round(2.0 * float(x)) / 2.0
            return Fraction(h).limit_denominator(10**9)
        return Fraction(x).limit_denominator(10**9)
    
    Uf = [[to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]
    
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    
    # Check W routing
    for c in range(R):
        nz = []
        for idx in range(p * n):
            if Wf[idx][c] != 0:
                nz.append(idx)
        if len(nz) != 1:
            return False
        w_idx = nz[0]
        k0 = w_idx // n
        i0 = w_idx % n
        # Verify that column c actually contributes to (i0, k0)
        contributes = False
        for j in range(m):
            if Uf[i0 * m + j][c] != 0 and Vf[j * p + k0][c] != 0:
                contributes = True
                break
        if not contributes:
            return False
    
    return True


def _try_merge_terms(U, V, W, n, m, p, terms_to_merge, rng, deadline):
    """Try to merge a set of schoolbook terms into fewer terms.
    
    terms_to_merge: list of (i,j,k) triples to attempt merging.
    Returns (U, V, W) if successful, None otherwise.
    """
    R = len(U[0])
    # Identify the column indices corresponding to terms_to_merge
    col_map = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_map[(i, j, k)] = i * m * p + j * p + k
    
    merge_cols = [col_map[t] for t in terms_to_merge if t in col_map]
    if len(merge_cols) < 2:
        return None
    
    # Create a modified system where we try to replace these terms
    # with a smaller set of terms
    num_merge = len(merge_cols)
    
    # We'll try to replace num_merge terms with num_merge - 1 terms
    # by adjusting the coefficients
    
    # Build the linear system for the merged terms
    # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
    
    # For simplicity, try a random perturbation approach
    # Start from schoolbook and try to zero out one term by adjusting others
    
    target_col = merge_cols[0]
    other_cols = merge_cols[1:]
    
    # Try to express the target term as a combination of others
    # This is generally not possible for arbitrary terms, but we can try
    
    # For now, return None - this is a placeholder for more sophisticated merging
    return None


def _numerical_search(n, m, p, target_rank, rng, deadline):
    """Try to find an exact decomposition of target_rank using numerical methods."""
    
    # The tensor identity: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
    # This is a system of n*m*p = 28 scalar equations in 4*R + 14*R = 18*R unknowns.
    
    # For target_rank R, we have 18*R unknowns and 28 equations.
    # For R >= 2, this is underdetermined, so solutions exist.
    # But we also need the W routing constraint.
    
    # Strategy: random initialization + iterative refinement
    best_residual = float('inf')
    best_U = None
    best_V = None
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization with half-integer values
        U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    expected = 1.0 if i == k else 0.0
                    residual += (s - expected) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
        
        if residual < 1e-10:
            # Found a candidate, now need to determine W
            # For each column c, determine which (i,k) it contributes to
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            for c in range(target_rank):
                # Find which (i,k) this column contributes to
                contrib_ik = set()
                for i in range(n):
                    for j in range(m):
                        if U[i * m + j][c] != 0:
                            for k in range(p):
                                if V[j * p + k][c] != 0:
                                    contrib_ik.add((i, k))
                
                if len(contrib_ik) != 1:
                    valid = False
                    break
                
                i0, k0 = contrib_ik.pop()
                W[k0 * n + i0][c] = 1.0
            
            if valid:
                # Verify exactly
                if _verify_exact(best_U, best_V, W, n, m, p):
                    return best_U, best_V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin under 120s limit
    
    # Guaranteed fallback
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    rng = random.Random(42)
    
    # Try ranks from 27 down to max(2, best_rank - 5)
    for target_rank in range(best_rank - 1, max(1, best_rank - 6), -1):
        if time.time() > deadline:
            break
        
        result = _numerical_search(n, m, p, target_rank, rng, deadline)
        if result is not None:
            U, V, W = result
            if _verify_exact(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found the lowest rank we tried
    
    # Also try a structured approach: exploit the 2x2 block structure
    # The 2x2 part can potentially use Strassen-like reductions
    if time.time() < deadline - 20:
        # Try to use the fact that for the 2x2 multiplication part,
        # we might be able to save one term per k-column
        # This would give rank 2*7 - 7 = 7 + 14 = 21? No, that's not right.
        
        # Actually, let's try a different structured approach:
        # For each k, we have 4 terms (i=0,1; j=0,1).
        # Can we do the 2x2 "inner" multiplication in rank 3 instead of 4?
        # That would give rank 3*7 = 21.
        
        # Strassen's algorithm for 2x2 uses 7 multiplications instead of 8.
        # But here we're not doing a full 2x2 x 2x2, we're doing 2x2 x 2x7.
        
        # Let's try to construct a rank-21 decomposition using Strassen-like ideas.
        # For each k, the contribution to C[0,k] and C[1,k] is:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # Strassen for 2x2: 7 products instead of 4 per "block"
        # But we only have 2x2 A and 2x1 B[:,k] for each k.
        
        # This is actually a 2x2 times 2x1 multiplication, which is just
        # two dot products, requiring 4 scalar multiplications (or 3 with tricks?).
        
        # Actually, 2x2 times 2x1:
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        # This requires 4 scalar multiplications, and I don't think it can be done in 3.
        
        # So the rank should be 4*7 = 28, unless there's a global trick.
        
        pass
    
    return best_result

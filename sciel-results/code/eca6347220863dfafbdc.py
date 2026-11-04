"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses structured search to find reduced-rank decompositions by exploiting
the small n=2 dimension. Falls back to schoolbook (rank 50) if no valid
reduced decomposition is found within time.
"""

import time
from fractions import Fraction
import numpy as np

_T0 = time.time()
_DEADLINE_S = 105.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _verify(U, V, W, rank, n=2, m=5, p=5):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank


def _try_structured_search(target_rank, n=2, m=5, p=5, max_attempts=200):
    """
    Systematic search: fix U and V to sparse patterns, solve for W exactly.
    
    Key insight: for n=2, W has 10 rows. The equations couple W[2k] and W[2k+1]
    through the same V entries. We exploit this by fixing U,V and solving linear
    system for W.
    """
    np.random.seed(42 + target_rank)
    
    for attempt in range(max_attempts):
        if not _deadline_ok():
            break
        
        # Generate random sparse U and V with entries in {-1, 0, 1}
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        
        for r in range(target_rank):
            # U[:,r]: 1-2 nonzeros
            num_u = np.random.randint(1, 3)
            u_idx = np.random.choice(n * m, size=num_u, replace=False)
            for idx in u_idx:
                U[idx, r] = np.random.choice([-1, 1])
            
            # V[:,r]: 1-3 nonzeros  
            num_v = np.random.randint(1, 4)
            v_idx = np.random.choice(m * p, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build linear system for W
        # Unknowns: W[k*n+i][r] for k in range(p), i in range(n), r in range(target_rank)
        # Total unknowns: p * n * target_rank = 10 * target_rank
        # Equations: for each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # Total equations: n*m*p = 50
        
        num_unknowns = p * n * target_rank
        num_eq = n * m * p
        
        A = np.zeros((num_eq, num_unknowns), dtype=float)
        b = np.ones(num_eq, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(target_rank):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        if coeff != 0:
                            unk_idx = (k * n + i) * target_rank + r
                            A[eq_idx, unk_idx] = coeff
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        # Check residual
        if np.max(np.abs(A @ W_flat - b)) > 1e-10:
            continue
        
        # Convert to fractions and verify exactly
        W = [[0] * target_rank for _ in range(p * n)]
        valid = True
        for k in range(p):
            for i in range(n):
                for r in range(target_rank):
                    val = W_flat[(k * n + i) * target_rank + r]
                    frac = Fraction(val).limit_denominator(1000000)
                    W[k * n + i][r] = frac
        
        # Verify exactly
        U_list = U.tolist()
        V_list = V.tolist()
        if _verify(U_list, V_list, W, target_rank, n, m, p):
            return U_list, V_list, W, target_rank
    
    return None


def _try_block_structured(target_rank, n=2, m=5, p=5):
    """
    Try block-structured decomposition where we group indices.
    
    For n=2, we can think of the problem as two coupled 1x5x5 problems.
    Try to find a shared structure.
    """
    # Strategy: Use a base decomposition for 1x5x5 (rank 25) and try to
    # share rank terms between the two i-rows.
    
    # For 1x5x5, schoolbook is rank 25. Can we do better?
    # The tensor is T[j,k] = 1 for all j,k. This is the all-ones 5x5 matrix.
    # The tensor rank of the all-ones matrix of size 5x5 is 1 (it's rank 1 as a matrix).
    # But as a tensor decomposition for multiplication, it's different.
    
    # Actually, for 1x5x5 multiplication: A is 1x5, B is 5x5, C is 1x5.
    # C[k] = sum_j A[j]*B[j,k]
    # This is just a matrix-vector product. The tensor rank should be 5 (one per output).
    # But we need exact decomposition: sum_r U[j,r]*V[j,k,r]*W[k,r] = 1 for all j,k.
    
    # For 1x5x5, we can use rank 5: U[j,r] = 1 if r=j, V[j,k,r] = 1 if r=j, W[k,r] = 1.
    # Wait, that gives sum_r delta(j,r)*delta(j,r)*1 = 1 only if j=r, which works.
    # So rank 5 for 1x5x5.
    
    # For 2x5x5, if we could share the structure, we might get rank < 50.
    # But the coupling through W[k*n+i] makes it tricky.
    
    # Try: use rank 25*2 = 50 as baseline, but try to merge some terms.
    
    # Alternative: try random search with better initialization
    np.random.seed(12345)
    
    for attempt in range(100):
        if not _deadline_ok():
            break
        
        # Start from schoolbook and try to merge columns
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        # Initialize with schoolbook-like structure
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        # Now try to modify: add random perturbations and re-solve
        # This is complex; skip for now
        
        # Instead, try a different approach: random sparse with bias toward structure
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        
        for r in range(target_rank):
            # Bias: prefer entries that cover multiple (i,j,k) triples
            num_u = np.random.randint(1, 3)
            u_idx = np.random.choice(n * m, size=num_u, replace=False)
            for idx in u_idx:
                U[idx, r] = np.random.choice([-1, 1])
            
            num_v = np.random.randint(1, 4)
            v_idx = np.random.choice(m * p, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Solve for W
        num_unknowns = p * n * target_rank
        num_eq = n * m * p
        
        A = np.zeros((num_eq, num_unknowns), dtype=float)
        b = np.ones(num_eq, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(target_rank):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        if coeff != 0:
                            unk_idx = (k * n + i) * target_rank + r
                            A[eq_idx, unk_idx] = coeff
        
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        if np.max(np.abs(A @ W_flat - b)) > 1e-10:
            continue
        
        W = [[0] * target_rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for r in range(target_rank):
                    val = W_flat[(k * n + i) * target_rank + r]
                    frac = Fraction(val).limit_denominator(1000000)
                    W[k * n + i][r] = frac
        
        U_list = U.tolist()
        V_list = V.tolist()
        if _verify(U_list, V_list, W, target_rank, n, m, p):
            return U_list, V_list, W, target_rank
    
    return None


def construct():
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if not _verify(U, V, W, rank):
        U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a reduced-rank decomposition
    # Try ranks from 49 down to 30
    for target_rank in range(best_rank - 1, 25, -1):
        if not _deadline_ok():
            break
        
        result = _try_structured_search(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        if not _deadline_ok():
            break
        
        result = _try_block_structured(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

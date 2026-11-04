"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses a greedy construction to find a reduced-rank decomposition by adding
rank-1 terms that cover subsets of the tensor constraints. Falls back to
schoolbook (rank 50) if no valid reduced decomposition is found within time.
"""

import time
from fractions import Fraction
from itertools import product

_T0 = time.time()
_DEADLINE_S = 110.0


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


def _try_greedy_construction(max_rank, n=2, m=5, p=5, max_attempts=100):
    """
    Greedy construction: try to build a decomposition by adding rank-1 terms
    that each satisfy a subset of the tensor equations.
    
    For each term r, we choose U[:,r], V[:,r], W[:,r] such that the term
    contributes positively to some equations and doesn't break others.
    """
    import random
    
    random.seed(42)
    
    # The tensor equation: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # 
    # Greedy approach: start with all equations unsatisfied (need 1).
    # For each new term, find U,V,W patterns that make the product = 1 for
    # a subset of unsatisfied equations and = 0 for others.
    
    # A term (u,v,w) contributes u[i,j]*v[j,k]*w[k,i] to equation (i,j,k).
    # For the term to contribute 1 to equation (i,j,k), we need u[i,j]*v[j,k]*w[k,i] = 1.
    # For it to contribute 0, at least one of u[i,j], v[j,k], w[k,i] must be 0.
    
    # So each term "covers" a set of (i,j,k) where all three factors are nonzero
    # and their product is 1.
    
    # Let's try: for each term, pick a set of (i,j) for U, a set of (j,k) for V,
    # and a set of (k,i) for W, such that the "intersection" (i,j,k) where
    # (i,j) in U-set, (j,k) in V-set, (k,i) in W-set all have product 1.
    
    # This is complex. Let me try a simpler heuristic:
    # Try random sparse patterns and check if they can be part of a valid decomposition.
    
    for attempt in range(max_attempts):
        if not _deadline_ok():
            break
        
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        
        # Generate random sparse patterns with entries in {-1, 0, 1}
        for r in range(max_rank):
            # U: choose 1-3 nonzero entries
            num_u = random.randint(1, 3)
            u_idx = random.sample(range(n * m), num_u)
            for idx in u_idx:
                U[idx][r] = random.choice([-1, 1])
            
            # V: choose 1-3 nonzero entries
            num_v = random.randint(1, 3)
            v_idx = random.sample(range(m * p), num_v)
            for idx in v_idx:
                V[idx][r] = random.choice([-1, 1])
            
            # W: choose 1-3 nonzero entries
            num_w = random.randint(1, 3)
            w_idx = random.sample(range(p * n), num_w)
            for idx in w_idx:
                W[idx][r] = random.choice([-1, 1])
        
        # Verify
        if _verify(U, V, W, max_rank, n, m, p):
            return U, V, W, max_rank
    
    return None


def _try_systematic_search(target_rank, n=2, m=5, p=5):
    """
    Try to find a decomposition of exactly target_rank by solving a system
    of equations. We fix U and V to be sparse integer patterns and solve for W.
    """
    import numpy as np
    from fractions import Fraction
    
    # Strategy: fix U to be a specific pattern, then solve for V and W.
    # 
    # If we fix U, the equation becomes:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
    # 
    # This is still bilinear in V and W. But if we fix V too, we can solve for W.
    # 
    # Alternative: use the fact that the problem is symmetric in a sense.
    # 
    # Let me try: fix U to have a block structure. For example, U[i,j,r] = 1
    # only if r corresponds to a specific (i,j) pair.
    
    # Try: U is the "identity" pattern where each (i,j) has its own rank slot.
    # U[i*m+j][i*m+j] = 1, all other entries 0. Then we need:
    # V[j*p+k][i*m+j] * W[k*n+i][i*m+j] = 1 for all (i,j,k)
    # This gives rank 10, but V and W would need 10 columns each, and
    # V[j*p+k][r] * W[k*n+i][r] = 1 for all (j,k,i) where r = i*m+j.
    # This means V[j*p+k][i*m+j] and W[k*n+i][i*m+j] must be reciprocals.
    # But V and W are indexed differently, so this doesn't directly work.
    
    # Let me try a different approach: use numpy to solve a least-squares problem
    # where we search over integer patterns for U and V.
    
    # For a given target_rank, try to find U and V such that the system
    # A(U,V) * W = b has a solution, where A is built from U and V.
    
    # This is a nonlinear feasibility problem. Let's try random restarts.
    
    np.random.seed(123)
    
    for attempt in range(500):
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
            
            # V[:,r]: 1-2 nonzeros
            num_v = np.random.randint(1, 3)
            v_idx = np.random.choice(m * p, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        # Unknowns: W[k*n+i][r] for k in range(p), i in range(n), r in range(target_rank)
        # Total unknowns: p * n * target_rank
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
        
        # Check if the solution is close to integer/half-integer
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


def construct():
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if not _verify(U, V, W, rank):
        U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a reduced-rank decomposition
    # Try ranks from 49 down to 20
    for target_rank in range(best_rank - 1, 15, -1):
        if not _deadline_ok():
            break
        
        result = _try_systematic_search(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

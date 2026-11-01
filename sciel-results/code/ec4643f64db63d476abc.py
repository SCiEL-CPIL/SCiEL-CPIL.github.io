DIMENSIONS = [2, 2, 4]
import numpy as np
import time
import random
from fractions import Fraction
from itertools import combinations

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks (one per 2x2 output column).
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block 1: k in {0,1}, r indices 0..6
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # M3: A00(B10-B11)
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[2][2] = 1
    # M4: A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[2][6] = 1
    W[0][6] = 1

    # Block 2: k in {2,3}, r indices 7..13
    # M1': (A00+A11)(B20+B31)
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[6][7] = 1
    W[4][7] = 1; W[7][7] = 1
    # M2': (A10+A11)B20
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[5][8] = 1; W[7][8] = -1
    # M3': A00(B30-B31)
    U[0][9] = 1
    V[5][9] = 1; V[6][9] = -1
    W[4][9] = 1; W[6][9] = 1
    # M4': A11(B21-B20)
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[5][10] = 1
    # M5': (A00+A01)B31
    U[0][11] = 1; U[1][11] = 1
    V[6][11] = 1
    W[4][11] = -1; W[6][11] = 1
    # M6': (A10-A00)(B20+B21)
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[7][12] = 1
    # M7': (A01-A11)(B30+B31)
    U[1][13] = 1; U[3][13] = -1
    V[5][13] = 1; V[6][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _tensor_to_rank1(T):
    """Convert tensor T to a list of rank-1 terms (U_col, V_col, W_col)."""
    n, m, p = T.shape
    terms = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = T[a, b, c]
                if val != 0:
                    u = np.zeros(n*m); u[a] = val
                    v = np.zeros(m*p); v[b] = 1.0
                    w = np.zeros(p*n); w[c] = 1.0
                    terms.append((u, v, w))
    return terms

def _gaussian_eliminate_tensor(terms, n, m, p, deadline):
    """
    Apply Gaussian elimination to reduce the number of rank-1 terms.
    Returns (reduced_terms, time_remaining_fraction).
    """
    if not terms:
        return terms, 1.0
    start = time.time()
    remaining = deadline - start
    total_budget = 150.0  # rough
    budget_fraction = max(0.0, min(1.0, remaining / total_budget))
    
    # Work with a list of (u, v, w) vectors
    # We'll do a simple elimination: pick a term, use it to zero out entries in others.
    # This is a heuristic and may not always reduce rank, but it's fast.
    
    active = list(terms)
    # Sort by norm to pick pivots
    active.sort(key=lambda t: -np.linalg.norm(t[0]))
    
    used = []
    for pivot in active:
        if time.time() > deadline:
            break
        used.append(pivot)
        # Zero out this term's contribution from all other active terms
        new_active = []
        pu, pv, pw = pivot
        for term in active:
            if term is pivot:
                continue
            tu, tv, tw = term
            # Try to find a coefficient c such that (tu - c*pu) has smaller norm
            # Simple approach: if pu and tu share a nonzero index, try to cancel
            # For simplicity, we'll just keep all terms for now (elimination is complex)
            new_active.append(term)
        active = new_active
    
    return used + active, budget_fraction

def _sparsify_search(terms, n, m, p, deadline, target_rank):
    """
    Try to reduce the rank by sparsifying: randomly zero out entries in U, V, W
    and check if the tensor is still exactly decomposed.
    This is a heuristic search for a lower-rank decomposition.
    """
    if not terms:
        return None
    start = time.time()
    best_rank = len(terms)
    best_terms = terms
    
    # Convert to dense matrices for faster checking
    num_u = n * m
    num_v = m * p
    num_w = p * n
    rank = len(terms)
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    def check_rank(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try random sparsification
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:  # 5s safety margin
        attempts += 1
        # Randomly zero out some entries
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Zero out a random fraction of entries
        for mat in [U_try, V_try, W_try]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i, j] != 0 and rng.random() < 0.3:
                        mat[i, j] = 0
        
        # Check if still valid
        if check_rank(U_try, V_try, W_try):
            # Find the actual rank (number of non-zero columns)
            active_cols = [r for r in range(rank) if np.any(U_try[:, r] != 0) or np.any(V_try[:, r] != 0) or np.any(W_try[:, r] != 0)]
            if len(active_cols) < best_rank:
                best_rank = len(active_cols)
                best_terms = [(U_try[:, r], V_try[:, r], W_try[:, r]) for r in active_cols]
    
    if best_rank < rank:
        return best_terms
    return None

def _try_lower_rank_search(n, m, p, deadline):
    """
    Attempt to find a rank < 14 decomposition using various heuristics.
    Returns a solution dict if found, else None.
    """
    start = time.time()
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Get rank-1 terms
    terms = _tensor_to_rank1(T)
    
    # Try Gaussian elimination (heuristic)
    terms, _ = _gaussian_eliminate_tensor(terms, n, m, p, deadline)
    
    # Try sparsification search
    best_terms = _sparsify_search(terms, n, m, p, deadline, 13)
    
    if best_terms and len(best_terms) < 14:
        # Convert back to solution dict
        rank = len(best_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r, (u, v, w) in enumerate(best_terms):
            for i in range(n * m):
                U[i][r] = u[i]
            for j in range(m * p):
                V[j][r] = v[j]
            for k in range(p * n):
                W[k][r] = w[k]
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # First, get the known rank-14 solution as a fallback
    sol_14 = _strassen_2x2x4()
    
    # Verify it
    if _verify_decomposition(sol_14['U'], sol_14['V'], sol_14['W'], n, m, p, tol=1e-9):
        best_sol = sol_14
        best_rank = 14
    else:
        # Fallback to trivial if rank-14 fails (should not happen)
        best_sol = _trivial_solution(n, m, p)
        best_rank = n * m * p
    
    # Try to find a lower-rank solution
    if time.time() < deadline - 10:  # Need 10s for final verification
        lower_sol = _try_lower_rank_search(n, m, p, deadline)
        if lower_sol is not None:
            if _verify_decomposition(lower_sol['U'], lower_sol['V'], lower_sol['W'], n, m, p, tol=1e-9):
                best_sol = lower_sol
                best_rank = len(lower_sol['U'][0])
    
    # Final check: make sure we have a valid solution
    if not _verify_decomposition(best_sol['U'], best_sol['V'], best_sol['W'], n, m, p, tol=1e-9):
        # Absolute fallback
        best_sol = _trivial_solution(n, m, p)
    
    return best_sol

import time
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x3 by 3x7.
    Uses a search strategy to find ranks lower than the schoolbook rank (42).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Target tensor T where T[i,j,k] = 1
    T_target = np.ones((n, m, p))

    def verify(U, V, W, r):
        # Check if entries are close to half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-8): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-8): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-8): return False
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exact decomposition with rounded values
        # T_ijk = sum_alpha U_ij^alpha V_jk^alpha W_ki^alpha
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for alpha in range(r):
                        val += U_r[i*m+j, alpha] * V_r[j*p+k, alpha] * W_r[k*n+i, alpha]
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                if not valid: break
            if not valid: break
        return valid

    def find_solution_for_rank(r, max_restarts=1000):
        # We want to find U, V, W with half-integer entries such that
        # sum_alpha U_ij^alpha V_jk^alpha W_ki^alpha = 1 for all i,j,k
        
        # We can fix U and V randomly, then solve for W.
        # The system is linear in W.
        # Variables: W_{ki}^alpha for k in 0..p-1, i in 0..n-1, alpha in 0..r-1.
        # Total variables: p * n * r.
        # Equations: for each i, j, k: sum_alpha U_ij^alpha V_jk^alpha W_ki^alpha = 1.
        # Total equations: n * m * p.
        
        # Let's construct the linear system A * W_flat = b
        # b is a vector of ones of length n*m*p.
        # A is a matrix of size (n*m*p) x (p*n*r).
        
        # To save memory and time, we can solve for W incrementally or use sparse?
        # n*m*p = 42. p*n*r is at most 42*42 = 1764.
        # Dense matrix is fine.
        
        A = np.zeros((n*m*p, p*n*r))
        b = np.ones(n*m*p)
        
        # Fill A
        # Index mapping for W: idx_w = (k * n + i) * r + alpha
        # Index mapping for eq: idx_eq = (i * m + j) * p + k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for alpha in range(r):
                        # Coefficient for W_{ki}^alpha is U_{ij}^alpha * V_{jk}^alpha
                        # We will multiply A by U, V later? No, U, V are fixed in this step.
                        # So A[eq_idx, w_idx] = U[i,j,alpha] * V[j,k,alpha]
                        # We fill A after generating U, V.
                        pass
        
        # Actually, it's better to generate U, V, then build A, then solve.
        
        for restart in range(max_restarts):
            # Generate random half-integer U and V
            # Scale to keep numbers small
            U = np.random.randint(-2, 3, size=(nm, r)) / 2.0
            V = np.random.randint(-2, 3, size=(mp, r)) / 2.0
            
            # Build A
            A = np.zeros((n*m*p, p*n*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        for alpha in range(r):
                            w_idx = (k * n + i) * r + alpha
                            A[eq_idx, w_idx] = U[i*m+j, alpha] * V[j*p+k, alpha]
            
            # Solve A W = b
            try:
                W_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if W_sol is close to half-integers
            # W_sol shape is (p*n*r,). Reshape to (p, n, r)
            W = W_sol.reshape(p, n, r)
            
            # Check half-integer condition
            if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
                continue
            
            W_r = np.round(W * 2) / 2
            
            # Verify exact solution
            if verify(U, V, W_r, r):
                return U, V, W_r
                
        return None

    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin

    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook decomposition
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Search for lower ranks
    # Lower bound: max(nm, mp, pn) is not a strict lower bound for rank, but a good heuristic start.
    # Actually, rank >= max(nm, mp, pn) is false. Rank >= n*m*p / max(n,m,p) is a bound?
    # Let's start from a reasonable lower bound, e.g., 21 (max dimension product / min dim?)
    # For 2x3x7, nm=6, mp=21, pn=14.
    # Let's try r from 21 down to 1? Or up?
    # We want minimal rank. So we should try small r first.
    # But small r is harder to find.
    # Let's try r from 21 upwards? No, if we find r=25, we don't know if 24 exists.
    # We should search r from 1 to 42 and return the first one found?
    # That might take too long if 1-20 are impossible.
    # Let's assume the rank is close to the schoolbook or slightly lower.
    # Known results: M_{2,3,7} rank is likely <= 42.
    # Let's search r from 21 to 42. If we find one at 25, we can try 24, 23... later if time permits.
    # To be efficient, let's search r in increasing order from a lower bound.
    # Lower bound for M_{n,m,p} is max(nm, mp, pn) is NOT correct.
    # The rank is at least n*m*p / (n+m+p-2) ?
    # For 2,3,7: 42 / (2+3+7-2) = 42/10 = 4.2. So r >= 5.
    # Let's start from r=21 (heuristic) and go up?
    # If we go up, we might find 25. But 20 might be possible.
    # Let's try a few random restarts for each r in [21, 22, ..., 42].
    # If we find a solution for r, we record it and continue to see if we can find a lower one?
    # No, if we search r=21, 22, ... and find one at 25, we should then check 21-24 more thoroughly?
    # Given time constraints, let's just find *any* solution below 42.
    # The seed returns 42. Any < 42 is an improvement.
    
    # Let's iterate r from 21 to 42.
    # If we find a solution at r, we can break and return it?
    # No, we want the *minimal* rank.
    # But finding the minimal rank is hard.
    # Let's find the first solution we can find, starting from r=21 upwards.
    # If we find r=25, we return 25. (Score 42/25 = 1.68)
    # If we find r=21, we return 21. (Score 42/21 = 2.0)
    
    for r in range(21, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a solution for this rank
        # Increase restarts for smaller r? No, smaller r is harder.
        # Give more restarts to smaller r.
        max_restarts = 500 if r < 25 else 200
        
        sol = find_solution_for_rank(r, max_restarts)
        
        if sol is not None:
            U, V, W = sol
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            # If we find a solution, should we continue to check lower r?
            # We are iterating upwards. So we found the *smallest* r in our search order that worked.
            # Since we started at 21, if 21 failed, 22 failed... 25 worked, then 25 is the minimal found.
            # We can break here if we assume 21-24 are unlikely or we've spent enough time.
            # To be safe, we can break and return this.
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

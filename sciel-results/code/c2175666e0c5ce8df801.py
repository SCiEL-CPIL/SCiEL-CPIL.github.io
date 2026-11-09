"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    x2 = 2 * x
    return abs(x2 - round(x2)) < 1e-9

def _check_all_valid(U, V, W):
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                if not _is_valid_entry(x):
                    return False
    return True

def _try_rank(r, n, m, p, deadline, max_iters=200):
    """Try to find a rank-r decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    best = None
    
    for it in range(max_iters):
        if time.time() > deadline - 15:
            break
        
        # Random initialization with small values
        scale = 1.0
        U = [[random.choice([-1, 0, 1, 0.5, -0.5, 1.5, -1.5]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1, 0.5, -0.5, 1.5, -1.5]) for _ in range(r)] for _ in range(mp)]
        
        # For each c, solve for W_c given U and V
        # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # For fixed U,V, this is linear in W.
        
        # Build linear system: for each (i,j,k), equation: sum_c A_c[i,j,k] * W_c[i,k] = 1
        # where A_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c]
        # Unknowns: W_c[i,k] for c in [0,r), i in [0,n), k in [0,p)
        # Total unknowns: r * n * p
        # Total equations: n * m * p
        
        num_unknowns = r * n * p
        num_equations = n * m * p
        
        if num_unknowns < num_equations:
            continue  # Underdetermined system unlikely to have solution
        
        # Build the system
        # Variable index: c * (n*p) + i * p + k
        A_mat = np.zeros((num_equations, num_unknowns))
        b_vec = np.ones(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        val = U[i*m+j][c] * V[j*p+k][c]
                        var_idx = c * (n*p) + i * p + k
                        A_mat[eq_idx, var_idx] = val
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is exact
        if len(residuals) > 0 and residuals[0] > 1e-6:
            continue
        
        # Check if entries are close to integer/half-integer
        W_mat = W_flat.reshape(r, n, p)
        W_list = [[0]*r for _ in range(pn)]
        valid = True
        for c in range(r):
            for i in range(n):
                for k in range(p):
                    w_val = W_mat[c, i, k]
                    # Snap to nearest half-integer
                    snapped = round(2*w_val) / 2.0
                    if abs(w_val - snapped) > 1e-6:
                        valid = False
                        break
                    W_list[k*n+i][c] = snapped
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify(U, V, W_list, n, m, p) and _check_all_valid(U, V, W_list):
            return {"U": U, "V": V, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known lower bound: at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But actual rank is likely higher. Try from 60 downward.
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank(r, n, m, p, deadline, max_iters=50)
        if result is not None:
            best = result
            best_rank = r
            break  # Found the minimal rank we can
    
    return best
# EVOLVE-BLOCK-END

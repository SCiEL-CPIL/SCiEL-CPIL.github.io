# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _search_233_fixed_solve(deadline):
    """Search for 2x3x3 decomposition by fixing U and V and solving for W."""
    n, m, p = 2, 3, 3
    best_rank = n * m * p
    best_solution = None
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
    ]
    
    # Try ranks from 12 upwards
    for r in range(12, best_rank):
        if time.time() > deadline:
            break
        trials = 5000
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            
            # Generate sparse U and V
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice(vs)
            
            # Build linear system for W
            # T_ijk = sum_c U_ic V_jc W_kc
            # For each (i, j, k), we have T_ijk = 1.
            # 1 = sum_c (U_ic V_jc) W_kc
            # This is a linear system in W_kc for fixed k? No, W is the unknown.
            # Let W be a matrix of shape (p*n, r).
            # The equation for each (i, j) is a vector of length p*n?
            # Actually, for each (i, j), we have p equations (one for each k).
            # 1 = sum_c (U_ic V_jc) W_{k, c}
            # This is linear in W_{k, c}.
            # We can stack all equations.
            # Total equations: n * m * p = 18.
            # Unknowns: p * n * r = 6 * r.
            
            # Construct matrix A and vector b
            # A has shape (n*m*p, p*n*r)
            # b has shape (n*m*p,)
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                        # Note: W index is k*n + i
                        for c in range(r):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            # W variable index: (k * n + i) * r + c
                            var_idx = (k * n + i) * r + c
                            A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W = sol.reshape(p * n, r)
            
            # Verify
            if _verify(U, V, W, n, m, p, r):
                best_rank = r
                best_solution = (U, V, W)
                # Found a valid one. Try to find lower rank?
                # We are iterating r ascending. So this is the minimum found so far.
                # We can break and return, as we want the lowest rank.
                return best_solution

    return best_solution

def _construct_from_233(U233, V233, W233, r233):
    """Construct 2x3x9 decomposition from 2x3x3."""
    n, m, p = 2, 3, 9
    R = 3 * r233
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for g in range(3):
        for c in range(r233):
            col = 3 * c + g
            for i in range(2):
                for j in range(3):
                    U[i * 3 + j, col] = U233[i, j, c]
            
            for j in range(3):
                for l in range(3):
                    k = 3 * g + l
                    V[j * 9 + k, col] = V233[j, l, c]
            
            for l in range(3):
                k = 3 * g + l
                for i in range(2):
                    W[k * 2 + i, col] = W233[l, i, c]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    sol_233 = _search_233_fixed_solve(deadline)
    if sol_233:
        U233, V233, W233 = sol_233
        r233 = U233.shape[1]
        candidate = _construct_from_233(U233, V233, W233, r233)
        r_candidate = 3 * r233
        if r_candidate < best_rank:
            best = candidate
            best_rank = r_candidate

    return best
# EVOLVE-BLOCK-END

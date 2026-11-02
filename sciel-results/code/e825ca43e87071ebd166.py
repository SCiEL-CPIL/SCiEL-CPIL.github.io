"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _generate_candidate_pool(n, m, p):
    """
    Generate a pool of candidate rank-1 tensors.
    Includes Strassen terms for 2x2 blocks and Schoolbook terms for all blocks.
    """
    candidates = []
    
    # 1. Strassen terms for 5 blocks of 2x2x2
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        def add_term(u_dict, v_dict, w_dict):
            U = np.zeros(n*m)
            V = np.zeros(m*p)
            W = np.zeros(p*n)
            for idx, val in u_dict.items(): U[idx] = val
            for idx, val in v_dict.items(): V[idx] = val
            for idx, val in w_dict.items(): W[idx] = val
            candidates.append((U, V, W))
            
        # M1
        add_term({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w11:1})
        # M2
        add_term({u10:1, u11:1}, {v00:1}, {w10:1, w11:-1})
        # M3
        add_term({u00:1}, {v01:1, v11:-1}, {w01:1, w11:1})
        # M4
        add_term({u11:1}, {v10:1, v00:-1}, {w00:1, w10:1})
        # M5
        add_term({u00:1, u01:1}, {v11:1}, {w01:1, w00:-1})
        # M6
        add_term({u10:1, u00:-1}, {v00:1, v01:1}, {w11:1})
        # M7
        add_term({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1})
        
        # Schoolbook for this 2x2x2 block
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    add_term({ui:1}, {vj:1}, {wk:1})

    # 2. Schoolbook for the last block (2x1x10)
    for i in range(n):
        for k in range(p):
            ui = i * m + 2
            vj = 2 * p + k
            wk = k * n + i
            U = np.zeros(n*m); V = np.zeros(m*p); W = np.zeros(p*n)
            U[ui] = 1; V[vj] = 1; W[wk] = 1
            candidates.append((U, V, W))
            
    return candidates

def _try_structured_absorption(n, m, p, time_budget):
    """
    Try to absorb schoolbook terms for the 2x1 block into Strassen terms
    by solving a linear system for coefficient adjustments.
    
    The idea: we have 35 Strassen terms (7 per 2x2 block x 5 blocks) and 
    20 schoolbook terms for the last 2x1x10 block. We check if some subset
    of the 20 schoolbook terms can be expressed as a linear combination of
    the 35 Strassen terms (with small integer coefficients), which would
    allow us to merge them and reduce rank.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Build the 35 Strassen tensor vectors
    strassen_terms = []
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        terms = [
            ({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w11:1}),
            ({u10:1, u11:1}, {v00:1}, {w10:1, w11:-1}),
            ({u00:1}, {v01:1, v11:-1}, {w01:1, w11:1}),
            ({u11:1}, {v10:1, v00:-1}, {w00:1, w10:1}),
            ({u00:1, u01:1}, {v11:1}, {w01:1, w00:-1}),
            ({u10:1, u00:-1}, {v00:1, v01:1}, {w11:1}),
            ({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1}),
        ]
        
        for u_dict, v_dict, w_dict in terms:
            U = np.zeros(n*m)
            V = np.zeros(m*p)
            W = np.zeros(p*n)
            for idx, val in u_dict.items(): U[idx] = val
            for idx, val in v_dict.items(): V[idx] = val
            for idx, val in w_dict.items(): W[idx] = val
            # Flatten the tensor
            tensor = np.zeros(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i*m*p + j*p + k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
            strassen_terms.append(tensor)
    
    # Build the 20 schoolbook tensor vectors for the 2x1x10 block
    schoolbook_terms = []
    for i in range(n):
        for k in range(p):
            ui = i * m + 2
            vj = 2 * p + k
            wk = k * n + i
            U = np.zeros(n*m)
            V = np.zeros(m*p)
            W = np.zeros(p*n)
            U[ui] = 1
            V[vj] = 1
            W[wk] = 1
            tensor = np.zeros(n*m*p)
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        tensor[ii*m*p + jj*p + kk] = U[ii*m+jj] * V[jj*p+kk] * W[kk*n+ii]
            schoolbook_terms.append(tensor)
    
    # Now try to find if any subset of schoolbook terms can be absorbed
    # We'll try all pairs of schoolbook terms to see if their sum/difference
    # can be expressed as a combination of Strassen terms
    
    S = np.array(strassen_terms).T  # shape (n*m*p, 35)
    SB = np.array(schoolbook_terms).T  # shape (n*m*p, 20)
    
    # For each pair of schoolbook terms, check if their sum is in span of Strassen
    for a in range(20):
        if time.time() > deadline:
            break
        for b in range(a+1, 20):
            if time.time() > deadline:
                break
            
            # Try: SB_a + SB_b = sum of some Strassen terms with small integer coeffs
            # This is a linear system: S @ x = SB_a + SB_b
            target = SB[:, a] + SB[:, b]
            
            # Try to solve with small integer coefficients
            # First check if it's in the span
            try:
                # Use least squares to find approximate solution
                x_approx, residuals, rank_S, sv = np.linalg.lstsq(S, target, rcond=None)
                
                # Check if residual is small
                if residuals and residuals[0] < 1e-10:
                    # Check if x_approx is close to integers
                    x_int = np.round(x_approx)
                    if np.allclose(S @ x_int, target, atol=1e-10):
                        # Found a valid combination!
                        # Now construct the new decomposition
                        # We replace SB_a and SB_b with the combination
                        non_zero = np.where(np.abs(x_int) > 0.5)[0]
                        if len(non_zero) < 2:  # Need fewer terms than the 2 we're replacing
                            # Build new U, V, W
                            U = np.zeros((n*m, 54))
                            V = np.zeros((m*p, 54))
                            W = np.zeros((p*n, 54))
                            
                            col = 0
                            # Add Strassen terms with coefficients
                            for si in non_zero:
                                coeff = int(x_int[si])
                                if coeff == 0:
                                    continue
                                # Get the original Strassen term
                                # Reconstruct from strassen_terms index
                                g = si // 7
                                t = si % 7
                                k0 = 2 * g
                                k1 = 2 * g + 1
                                
                                u00, u01, u10, u11 = 0, 1, 3, 4
                                v00, v01 = k0, k1
                                v10, v11 = 10 + k0, 10 + k1
                                w00, w01 = k0 * 2, k1 * 2
                                w10, w11 = k0 * 2 + 1, k1 * 2 + 1
                                
                                terms = [
                                    ({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w11:1}),
                                    ({u10:1, u11:1}, {v00:1}, {w10:1, w11:-1}),
                                    ({u00:1}, {v01:1, v11:-1}, {w01:1, w11:1}),
                                    ({u11:1}, {v10:1, v00:-1}, {w00:1, w10:1}),
                                    ({u00:1, u01:1}, {v11:1}, {w01:1, w00:-1}),
                                    ({u10:1, u00:-1}, {v00:1, v01:1}, {w11:1}),
                                    ({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1}),
                                ]
                                
                                u_dict, v_dict, w_dict = terms[t]
                                for idx, val in u_dict.items(): U[idx, col] = val * coeff
                                for idx, val in v_dict.items(): V[idx, col] = val * coeff
                                for idx, val in w_dict.items(): W[idx, col] = val * coeff
                                col += 1
                            
                            # Add remaining Strassen terms (those not in non_zero)
                            for si in range(35):
                                if si not in non_zero:
                                    g = si // 7
                                    t = si % 7
                                    k0 = 2 * g
                                    k1 = 2 * g + 1
                                    
                                    u00, u01, u10, u11 = 0, 1, 3, 4
                                    v00, v01 = k0, k1
                                    v10, v11 = 10 + k0, 10 + k1
                                    w00, w01 = k0 * 2, k1 * 2
                                    w10, w11 = k0 * 2 + 1, k1 * 2 + 1
                                    
                                    terms = [
                                        ({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w11:1}),
                                        ({u10:1, u11:1}, {v00:1}, {w10:1, w11:-1}),
                                        ({u00:1}, {v01:1, v11:-1}, {w01:1, w11:1}),
                                        ({u11:1}, {v10:1, v00:-1}, {w00:1, w10:1}),
                                        ({u00:1, u01:1}, {v11:1}, {w01:1, w00:-1}),
                                        ({u10:1, u00:-1}, {v00:1, v01:1}, {w11:1}),
                                        ({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1}),
                                    ]
                                    
                                    u_dict, v_dict, w_dict = terms[t]
                                    for idx, val in u_dict.items(): U[idx, col] = val
                                    for idx, val in v_dict.items(): V[idx, col] = val
                                    for idx, val in w_dict.items(): W[idx, col] = val
                                    col += 1
                            
                            # Add remaining schoolbook terms (all except a and b)
                            for si in range(20):
                                if si != a and si != b:
                                    i = si // p
                                    k = si % p
                                    ui = i * m + 2
                                    vj = 2 * p + k
                                    wk = k * n + i
                                    U[ui, col] = 1
                                    V[vj, col] = 1
                                    W[wk, col] = 1
                                    col += 1
                            
                            if col <= 54:
                                if _verify_np(U, V, W, n, m, p, tol=1e-8):
                                    return {
                                        "U": U.tolist(),
                                        "V": V.tolist(),
                                        "W": W.tolist()
                                    }
            except Exception:
                pass
    
    return None

def _try_milp_rank54(n, m, p, time_budget):
    """
    Use MIQP to find a sparse subset of candidate tensors that sums to the target tensor.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Construct target tensor
    T_target = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m*p + j*p + k] = 1

    candidates = _generate_candidate_pool(n, m, p)
    num_cands = len(candidates)
    
    # Build matrix A where A[:, c] is the flattened tensor for candidate c
    A = np.zeros((n * m * p, num_cands))
    for c, (U, V, W) in enumerate(candidates):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    A[idx, c] = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    
    # We want to find x in {0, 1} such that A @ x = T_target
    # Minimize sum(x)
    
    lb = np.zeros(num_cands)
    ub = np.ones(num_cands)
    
    try:
        res = milp(
            c=np.ones(num_cands),
            constraints=LinearConstraint(A, T_target, T_target),
            integrality=np.ones(num_cands),
            bounds=Bounds(lb, ub)
        )
        
        if time.time() > deadline:
            return None
            
        if res.status == 0: # Optimal
            x = res.x
            if np.sum(x) <= 54:
                rank = int(np.sum(x))
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                col = 0
                for c_idx in range(num_cands):
                    if x[c_idx] > 0.5:
                        Uc, Vc, Wc = candidates[c_idx]
                        U[:, col] = Uc
                        V[:, col] = Vc
                        W[:, col] = Wc
                        col += 1
                        
                if _verify_np(U, V, W, n, m, p, tol=1e-8):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try the rank-54 search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 5:
            # First try structured absorption (faster)
            result = _try_structured_absorption(n, m, p, time_left / 2)
            if result is not None:
                # Verify with Fractions to be absolutely sure
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = 54
            else:
                # Fall back to MILP
                time_left = deadline - time.time()
                if time_left > 5:
                    result = _try_milp_rank54(n, m, p, time_left)
                    if result is not None:
                        # Verify with Fractions to be absolutely sure
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = 54
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best

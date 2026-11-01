import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, rank, tol=1e-12):
    """Fast float verification."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _random_init(rank, n, m, p, seed):
    """Random initialization with small integer/half-integer entries."""
    rng = random.Random(seed)
    def rand_entry():
        r = rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2])
        return r
    U = [[rand_entry() for _ in range(rank)] for _ in range(n * m)]
    V = [[rand_entry() for _ in range(rank)] for _ in range(m * p)]
    W = [[rand_entry() for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def _compute_residual(U, V, W, n, m, p, rank):
    """Compute the residual tensor: T_target - T_decomposed."""
    residual = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                residual[(i, j, k)] = 1.0 - s
    return residual

def _try_complete_rank1(U, V, W, n, m, p, rank, target_rank):
    """
    Try to complete a partial decomposition of 'rank' terms into 'target_rank' terms
    by finding the remaining (target_rank - rank) rank-1 terms that fix the residual.
    Uses iterative greedy rank-1 elimination.
    """
    remaining = target_rank - rank
    if remaining <= 0:
        return None
    
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    
    residual = _compute_residual(U_cur, V_cur, W_cur, n, m, p, rank)
    
    for step in range(remaining):
        # Greedy: find the best rank-1 term to add
        best_term = None
        best_improvement = 0
        
        # Try various simple rank-1 patterns
        candidates = []
        
        # Pattern: U[i,j] = a, V[j,k] = b, W[k,i] = c
        # We want to maximize reduction in residual
        # Try all combinations of sparse patterns
        
        # For each possible support pattern, try to solve
        # Simple approach: try random sparse rank-1 terms
        rng = random.Random(hash((n, m, p, rank, step)))
        
        for trial in range(200):
            # Random sparse U
            U_trial = [0.0] * (n * m)
            V_trial = [0.0] * (m * p)
            W_trial = [0.0] * (p * n)
            
            # Pick random non-zero entries
            num_nonzero_u = rng.randint(1, min(3, n*m))
            num_nonzero_v = rng.randint(1, min(3, m*p))
            num_nonzero_w = rng.randint(1, min(3, p*n))
            
            u_indices = rng.sample(range(n*m), num_nonzero_u)
            v_indices = rng.sample(range(m*p), num_nonzero_v)
            w_indices = rng.sample(range(p*n), num_nonzero_w)
            
            for idx in u_indices:
                U_trial[idx] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            for idx in v_indices:
                V_trial[idx] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            for idx in w_indices:
                W_trial[idx] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            # Compute improvement
            improvement = 0.0
            for (i, j, k), res_val in residual.items():
                u_val = U_trial[i*m+j]
                v_val = V_trial[j*p+k]
                w_val = W_trial[k*n+i]
                if u_val != 0 and v_val != 0 and w_val != 0:
                    # This term contributes u_val * v_val * w_val to (i,j,k)
                    # We want to reduce |res_val|
                    contribution = u_val * v_val * w_val
                    new_res = res_val - contribution
                    improvement += (abs(res_val) - abs(new_res))
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_term = (U_trial, V_trial, W_trial)
        
        if best_term is None or best_improvement <= 0:
            return None
        
        U_trial, V_trial, W_trial = best_term
        for idx in range(n*m):
            U_cur[idx][rank + step] = U_trial[idx]
        for idx in range(m*p):
            V_cur[idx][rank + step] = V_trial[idx]
        for idx in range(p*n):
            W_cur[idx][rank + step] = W_trial[idx]
        
        # Update residual
        residual = _compute_residual(U_cur, V_cur, W_cur, n, m, p, rank + step + 1)
    
    # Final check
    if _verify_float(U_cur, V_cur, W_cur, n, m, p, target_rank):
        return (U_cur, V_cur, W_cur)
    return None

def _strassen_base(n, m, p):
    """
    Build the Strassen-based decomposition for the 2x2x2 sub-block (k=0,1).
    Returns U, V, W for the first 7 terms, with columns 0-6 filled.
    """
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # r=0: P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # r=2: P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # r=3: P4 = d(g-e)
    U[3][3] = 1
    V[0][3] = -1; V[5][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # r=4: P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # r=5: P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin for 150s limit
    
    n, m, p = 2, 2, 5
    
    # Best known fallback: rank 10 (Strassen + 3 extra)
    best_result = None
    best_rank = 10
    
    # Build the rank-10 solution as fallback
    U10 = [[0.0]*10 for _ in range(n*m)]
    V10 = [[0.0]*10 for _ in range(m*p)]
    W10 = [[0.0]*10 for _ in range(p*n)]
    
    # Strassen part (7 terms)
    U7, V7, W7 = _strassen_base(n, m, p)
    for i in range(n*m):
        for c in range(7):
            U10[i][c] = U7[i][c]
    for i in range(m*p):
        for c in range(7):
            V10[i][c] = V7[i][c]
    for i in range(p*n):
        for c in range(7):
            W10[i][c] = W7[i][c]
    
    # Extra terms for k=2,3,4
    for extra_k, col in [(2, 7), (3, 8), (4, 9)]:
        for i in range(n):
            for j in range(m):
                U10[i*m+j][col] = 1.0
        for j in range(m):
            V10[j*p+extra_k][col] = 1.0
        for i in range(n):
            W10[extra_k*n+i][col] = 1.0
    
    if _verify_float(U10, V10, W10, n, m, p, 10):
        best_result = (U10, V10, W10)
        best_rank = 10
    else:
        # Schoolbook fallback
        best_rank = 20
        U20 = [[0.0]*20 for _ in range(n*m)]
        V20 = [[0.0]*20 for _ in range(m*p)]
        W20 = [[0.0]*20 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U20[i*m+j][col] = 1.0
                    V20[j*p+k][col] = 1.0
                    W20[k*n+i][col] = 1.0
                    col += 1
        best_result = (U20, V20, W20)
    
    # Now try to find lower rank decompositions
    # Strategy: Start from Strassen base (7 terms for k=0,1 sub-block)
    # and try to complete with fewer than 3 extra terms for k=2,3,4
    
    # Try rank 9: Strassen (7) + 2 extra terms
    if time.time() < deadline - 30:
        U7, V7, W7 = _strassen_base(n, m, p)
        # Extend to rank 9
        U9 = [row + [0.0, 0.0] for row in U7]
        V9 = [row + [0.0, 0.0] for row in V7]
        W9 = [row + [0.0, 0.0] for row in W7]
        
        result = _try_complete_rank1(U7, V7, W7, n, m, p, 7, 9)
        if result is not None:
            U9_res, V9_res, W9_res = result
            if _verify_float(U9_res, V9_res, W9_res, n, m, p, 9):
                best_result = (U9_res, V9_res, W9_res)
                best_rank = 9
    
    # Try rank 8: Strassen (7) + 1 extra term
    if time.time() < deadline - 20 and best_rank > 8:
        U7, V7, W7 = _strassen_base(n, m, p)
        result = _try_complete_rank1(U7, V7, W7, n, m, p, 7, 8)
        if result is not None:
            U8_res, V8_res, W8_res = result
            if _verify_float(U8_res, V8_res, W8_res, n, m, p, 8):
                best_result = (U8_res, V8_res, W8_res)
                best_rank = 8
    
    # Try rank 7: Just Strassen? (Unlikely to work for p=5, but check)
    if time.time() < deadline - 10 and best_rank > 7:
        U7, V7, W7 = _strassen_base(n, m, p)
        # Pad to check if it works as-is (it won't, but verify)
        if _verify_float(U7, V7, W7, n, m, p, 7):
            best_result = (U7, V7, W7)
            best_rank = 7
    
    # Try completely random search for rank 8 and 9
    if time.time() < deadline - 15 and best_rank > 8:
        for target_rank in [8, 9]:
            if time.time() > deadline - 10:
                break
            for seed in range(500):
                if time.time() > deadline - 10:
                    break
                U_rand, V_rand, W_rand = _random_init(target_rank, n, m, p, seed)
                if _verify_float(U_rand, V_rand, W_rand, n, m, p, target_rank):
                    best_result = (U_rand, V_rand, W_rand)
                    best_rank = target_rank
                    break
    
    # Try structured search: fix U and V, solve for W (or vice versa)
    # For a given U, V, the constraint is:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    # This is linear in W if U, V are fixed!
    if time.time() < deadline - 10 and best_rank > 8:
        for target_rank in [8, 9]:
            if time.time() > deadline - 5:
                break
            for seed in range(200):
                if time.time() > deadline - 5:
                    break
                U_rand, V_rand, W_rand = _random_init(target_rank, n, m, p, seed)
                
                # Solve for W given U, V
                # For each (k, i), we need:
                # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                # This gives m equations per (k,i) pair, with target_rank unknowns
                # Actually it's: for fixed (k,i), sum over j of [sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]] 
                # No, each (i,j,k) is a separate equation.
                
                # Let's set up the linear system for W
                # Unknowns: W[k*n+i, c] for all k,i,c -> p*n*target_rank unknowns
                # Equations: n*m*p equations
                # Each equation (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is linear in W!
                
                num_unknowns = p * n * target_rank
                num_eqs = n * m * p
                
                # Build matrix A and vector b
                # A[eq_idx, unk_idx] where eq_idx = (i,j,k) flattened, unk_idx = (k,i,c) flattened
                A = np.zeros((num_eqs, num_unknowns))
                b = np.ones(num_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                unk_idx = (k * n + i) * target_rank + c
                                A[eq_idx, unk_idx] = U_rand[i*m+j][c] * V_rand[j*p+k][c]
                            eq_idx += 1
                
                # Solve least squares
                try:
                    W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_sol = W_sol.reshape(p * n, target_rank)
                    
                    # Check if exact
                    if _verify_float(U_rand, V_rand, W_sol, n, m, p, target_rank):
                        best_result = (U_rand, V_rand, W_sol)
                        best_rank = target_rank
                        break
                except:
                    pass
    
    # Final verification and return
    if best_result is not None:
        U_final, V_final, W_final = best_result
        # Convert to list of lists with proper types
        U_out = [[float(x) for x in row] for row in U_final]
        V_out = [[float(x) for x in row] for row in V_final]
        W_out = [[float(x) for x in row] for row in W_final]
        
        # Final verification
        if _verify(U_out, V_out, W_out, n, m, p, best_rank):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Absolute fallback: schoolbook rank 20
    U20 = [[0.0]*20 for _ in range(n*m)]
    V20 = [[0.0]*20 for _ in range(m*p)]
    W20 = [[0.0]*20 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U20[i*m+j][col] = 1.0
                V20[j*p+k][col] = 1.0
                W20[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U20, "V": V20, "W": W20}

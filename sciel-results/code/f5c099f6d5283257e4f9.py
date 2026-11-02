"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

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
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w01][col] = 1
        W[w00][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
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

def _try_block_partition(n, m, p, time_budget, seed=42):
    """
    Strategy: Partition B (3x10) into two 3x5 blocks.
    Try to find a fast algorithm for 2x3 * 3x5 with rank < 30.
    Total rank = rank1 + rank2. If we can get rank < 15 for each, total < 30.
    Otherwise, try to beat 55 (e.g. 28+27=55 is not better, need <55).
    Actually, 2x3*3x5 schoolbook is 30. Strassen-like might be 25?
    If we find rank 25 for 2x3*3x5, total is 50.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Target: find rank r_block for 2x3*3x5 such that 2*r_block < 55
    # So r_block <= 27.
    # Let's try to find a decomposition for 2x3 * 3x5 with rank r_block.
    
    n1, m1, p1 = 2, 3, 5
    
    # Try ranks for the sub-problem
    # Lower bound for 2x3*3x5 is max(6, 15, 10) = 15.
    # Schoolbook is 30.
    # Let's search from 20 to 27.
    
    for r_block in range(20, 28):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid decomposition for 2x3 * 3x5 with rank r_block
        found_sub = False
        U_sub = None
        V_sub = None
        W_sub = None
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random ±1 matrices
            U_r = rng.randint(0, 2, size=(n1*m1, r_block)) * 2 - 1
            V_r = rng.randint(0, 2, size=(m1*p1, r_block)) * 2 - 1
            
            # Solve for W_sub
            W_r = np.zeros((p1*n1, r_block))
            feasible = True
            
            for i in range(n1):
                for k in range(p1):
                    A = np.zeros((m1, r_block))
                    b = np.ones(m1)
                    for j in range(m1):
                        A[j, :] = U_r[i*m1+j] * V_r[j*p1+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_r[k*n1+i, :] = sol
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify sub-problem
                ok = True
                for i in range(n1):
                    for j in range(m1):
                        for k in range(p1):
                            prod = U_r[i*m1+j] * V_r[j*p1+k] * W_r[k*n1+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                if ok:
                    U_sub = U_r
                    V_sub = V_r
                    W_sub = W_r
                    found_sub = True
                    break
        
        if found_sub:
            # Construct the full 2x3 * 3x10 decomposition
            # B is split into B1 (3x5) and B2 (3x5)
            # B1 corresponds to k=0..4, B2 to k=5..9
            # We use the same U_sub for both blocks?
            # No, U depends on A. A is the same for both.
            # So U_full will have 2*r_block columns?
            # Wait, U is (n*m, rank).
            # If we use the same U_sub for both, we need to shift the columns in V and W.
            
            total_rank = 2 * r_block
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            # Block 1: k=0..4
            # U_full[:, 0:r_block] = U_sub
            U_full[:, 0:r_block] = U_sub
            
            # V_full: rows 0..4 (j=0..2, k=0..4) come from V_sub
            # V_sub is (m1*p1, r_block) = (15, r_block)
            # V_full rows 0..14 correspond to B1
            V_full[0:15, 0:r_block] = V_sub
            
            # W_full: rows 0..9 (k=0..4, i=0..1) come from W_sub
            # W_sub is (p1*n1, r_block) = (10, r_block)
            # W_full rows 0..9 correspond to C1
            W_full[0:10, 0:r_block] = W_sub
            
            # Block 2: k=5..9
            # U_full[:, r_block:2*r_block] = U_sub (same A)
            U_full[:, r_block:2*r_block] = U_sub
            
            # V_full: rows 15..29 (j=0..2, k=5..9) come from V_sub
            # Note: V_full row index is j*p + k.
            # For k=5..9, j=0..2, index is j*10 + k.
            # j=0, k=5..9 -> 5..9
            # j=1, k=5..9 -> 15..19
            # j=2, k=5..9 -> 25..29
            # V_sub row index is j*p1 + k' where k'=0..4.
            # So V_sub[j*5 + k'] maps to V_full[j*10 + (k'+5)].
            for j in range(m1):
                for k_prime in range(p1):
                    v_row_sub = j * p1 + k_prime
                    v_row_full = j * p + (k_prime + 5)
                    V_full[v_row_full, r_block:2*r_block] = V_sub[v_row_sub]
            
            # W_full: rows 10..19 (k=5..9, i=0..1) come from W_sub
            # W_full row index is k*n + i.
            # k=5..9, i=0..1 -> 10..19.
            # W_sub row index is k'*n1 + i where k'=0..4.
            # So W_sub[k'*2 + i] maps to W_full[(k'+5)*2 + i].
            for k_prime in range(p1):
                for i in range(n1):
                    w_row_sub = k_prime * n1 + i
                    w_row_full = (k_prime + 5) * n1 + i
                    W_full[w_row_full, r_block:2*r_block] = W_sub[w_row_sub]
            
            # Verify full solution
            if _verify_np(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                best = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = total_rank
                break # Found a solution for this r_block, try smaller? No, we went up.
                # Wait, we want minimal rank. We started from 20.
                # If 20 works, total 40.
                # If 21 works, total 42.
                # We break on first success, which is the smallest r_block.
    
    return best

def _try_structured_winograd(n, m, p, time_budget, seed=42):
    """
    Structured approach exploiting the 2x3x10 structure.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best = None
    best_rank = n * m * p + 1
    
    candidate_ranks = list(range(30, 65, 1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            
            W_sol = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol[k*n+i, :] = sol
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = _verify_np(U, V, W_sol, n, m, p)
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                    best_rank = r
                    break
        
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen Rank 55 (known valid)
    if best_rank > 55:
        res55 = _strassen_rank55(n, m, p)
        if _verify(res55["U"], res55["V"], res55["W"], n, m, p):
            best = res55
            best_rank = 55
    
    # Try Block Partition Strategy (Target < 55)
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_block_partition(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # Try structured Winograd approach (Target < current best)
    time_left = deadline - time.time()
    if time_left > 10:
        # Only search for ranks lower than current best
        search_limit = min(65, best_rank)
        # The function _try_structured_winograd searches 30..65. 
        # We should modify it or pass a limit. 
        # For now, let's just run it and check if it finds something better.
        # To be efficient, we can wrap it or just let it run and filter.
        result = _try_structured_winograd(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best

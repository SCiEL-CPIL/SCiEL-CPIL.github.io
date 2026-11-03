import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 115  # seconds, safety margin before 150s kill

    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if val != 1:
                        return False
        return True

    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_list(x) for x in obj]
        else:
            return obj

    def _build_A(U_np, V_np, R):
        """Build the linear system A @ W_flat = b for given U, V."""
        num_vars = p * n * R
        A_mat = np.zeros((n * m * p, num_vars))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_base = k * n + i
                    for r in range(R):
                        var_idx = w_base * R + r
                        A_mat[eq_idx, var_idx] = U_np[u_idx, r] * V_np[v_idx, r]
                    eq_idx += 1
        return A_mat

    def _try_solve_linear(U_np, V_np, R):
        """Try to solve for W given U, V. Returns W_np or None."""
        A_mat = _build_A(U_np, V_np, R)
        b_vec = np.ones(n * m * p)
        
        # Try to find a solution. We prefer integer or half-integer solutions.
        # First, try to find the minimum norm solution.
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check if W_flat is close to integers
            W_int = np.round(W_flat)
            if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                return W_int.reshape((p * n, R))
            
            # Check if W_flat is close to half-integers
            W_half = 2 * np.round(W_flat * 2) / 2
            if np.allclose(A_mat @ W_half, b_vec, atol=1e-9):
                return W_half.reshape((p * n, R))
                
        except:
            pass
            
        return None

    def _local_search(U_np, V_np, W_np, R, max_iter=500):
        """Local search to refine W (and potentially U, V) to exact integer/half-integer solution."""
        # We fix U and V for now and only adjust W, as W is the free variable in the linear system.
        # However, the linear system might be underdetermined, allowing us to move in the null space.
        # A simpler approach: if the linear solution was not exact, try to snap to nearest valid lattice point
        # by checking neighbors? No, that's expensive.
        
        # Instead, let's try a different strategy:
        # If lstsq gave a solution, but it wasn't integer/half-integer, maybe we can find a *different*
        # solution in the affine space that IS integer/half-integer.
        # This is a hard problem.
        
        # Let's stick to the linear algebra approach but try more random U, V.
        # Or, try to solve the system exactly using rational arithmetic? Too slow.
        
        # Let's try a "snapping" heuristic:
        # If the residual is small, try to round W to nearest half-integer and check validity.
        # If not valid, try rounding to nearest integer.
        
        W_int = np.round(W_np)
        if _verify_float(U_np, V_np, W_int, n, m, p):
            return W_int
        
        W_half = 2 * np.round(W_np * 2) / 2
        if _verify_float(U_np, V_np, W_half, n, m, p):
            return W_half
            
        return None

    def _verify_float(U_np, V_np, W_np, n, m, p):
        """Fast float verification."""
        R = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        s += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy 1: Random search for U, V and solve for W linearly
    # We iterate R from low to high? Or high to low?
    # Lower R is harder. Let's try R from 12 to 35.
    # Actually, the current code tries from 35 down to 21.
    # Let's try a mix. Start with some lower ranks that are theoretically possible?
    # The lower bound is max(n*m, m*p, p*n) = 12.
    
    # Let's try R from 12 up to 35, with more attempts for lower R?
    # Or just keep the original order but improve the solver.
    
    # The "dual space" inspiration suggests looking at the structure.
    # But for now, let's stick to the linear solve approach which is robust.
    
    # We will try each R from 12 to 35.
    # For each R, we do a number of random restarts.
    
    for R in range(12, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # Number of attempts
        # Give more attempts to lower ranks?
        # Let's say 200 attempts per rank.
        max_attempts = 200
        attempts = 0
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Sometimes use {-2, -1, 0, 1, 2} to allow more flexibility?
            # Let's stick to {-1, 0, 1} first.
            U_np = np.random.randint(-1, 2, size=(n * m, R))
            V_np = np.random.randint(-1, 2, size=(m * p, R))
            
            W_np = _try_solve_linear(U_np, V_np, R)
            
            if W_np is not None:
                # Verify exactly with Fractions
                U_list = [[Fraction(x) for x in row] for row in U_np.tolist()]
                V_list = [[Fraction(x) for x in row] for row in V_np.tolist()]
                W_list = [[Fraction(x) for x in row] for row in W_np.tolist()]
                
                if is_valid(U_list, V_list, W_list):
                    return {
                        "U": to_list(U_list),
                        "V": to_list(V_list),
                        "W": to_list(W_list)
                    }
    
    # Fallback to Schoolbook
    rank = rank_schoolbook
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
    return {"U": U, "V": V, "W": W}

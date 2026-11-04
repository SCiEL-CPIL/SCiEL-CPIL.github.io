"""
Strategy: Discrete Simulated Annealing on Rational Coefficients.

Instead of continuous optimization (L-BFGS), we treat the coefficients as variables 
taking values from a discrete set of rationals (0, +/-0.5, +/-1, +/-1.5, +/-2). 
We use Simulated Annealing to minimize the squared error of the tensor decomposition. 
If the error becomes sufficiently small, we attempt to reconstruct the exact rational 
solution and verify it. This approach is structurally different from gradient-based 
methods as it performs stochastic jumps in the discrete space, which can help escape 
local minima and directly target the required integer/half-integer constraints.
"""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """Check if given factor matrices produce correct matmul tensor (vectorized)."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    R = U_np.shape[1]
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    
    # Check zero entries
    for a in range(n*m):
        i_a = a // m
        j_a = a % m
        for b in range(m*p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    if abs(T[a, b, c]) > tol:
                        return False
    
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _compute_error(U_flat, V_flat, W_flat, R, n, m, p, target_indices, zero_indices):
    """Compute sum of squared errors for a given flat representation."""
    U_np = U_flat.reshape(n*m, R)
    V_np = V_flat.reshape(m*p, R)
    W_np = W_flat.reshape(p*n, R)
    
    err = 0.0
    # Non-zero entries
    for (a, b, c) in target_indices:
        val = np.dot(U_np[a], V_np[b] * W_np[c])
        err += (val - 1.0) ** 2
        
    # Zero entries
    for (a, b, c) in zero_indices:
        val = np.dot(U_np[a], V_np[b] * W_np[c])
        err += val ** 2
        
    return err

def _try_discrete_search(R, n, m, p, time_budget=10):
    """
    Attempt to find a rank-R decomposition using Simulated Annealing 
    on a discrete set of coefficients.
    """
    start_time = time.time()
    
    # Precompute indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
                
    zero_indices = []
    for a in range(n*m):
        i_a = a // m
        j_a = a % m
        for b in range(m*p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    zero_indices.append((a, b, c))
                    
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    total_vars = U_size + V_size + W_size
    
    # Candidate values for coefficients
    # Focus on small rationals
    candidates = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
    
    def _extract_and_verify(U_flat, V_flat, W_flat):
        U_np = U_flat.reshape(n*m, R)
        V_np = V_flat.reshape(m*p, R)
        W_np = W_flat.reshape(p*n, R)
        
        # Attempt rational reconstruction
        # If error is low, the values should be close to rationals with small denominators
        U_list = []
        V_list = []
        W_list = []
        
        # We will just use the floats for verification first. 
        # The task allows integer/half-integer. 
        # If the SA converges to a low error, the floats are likely very close to such values.
        # We can snap them.
        
        def snap(x):
            # Snap to nearest multiple of 0.5
            snapped = round(x * 2.0) / 2.0
            return snapped

        U_snapped = np.array([[snap(x) for x in row] for row in U_np])
        V_snapped = np.array([[snap(x) for x in row] for row in V_np])
        W_snapped = np.array([[snap(x) for x in row] for row in W_np])
        
        # Verify snapped version
        if _verify_rank(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p, tol=1e-8):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            
        # If snapping fails, try original floats (in case they are not exactly half-integers but still valid? 
        # Task says "integer/half-integer entries". So we must return those.
        # If the float solution is valid but not half-integer, it's not a valid solution for the contract 
        # unless we can find a half-integer one. 
        # However, if the error is 0, it might be a rational with larger denominator. 
        # But the contract specifies integer/half-integer.
        # So we only accept if the snapped version works.
        return None

    # Initialize
    # Start with a random sparse initialization
    random.seed(42)
    np.random.seed(42)
    
    x = np.zeros(total_vars)
    # Randomly assign some values
    for _ in range(total_vars // 5): # 20% non-zero
        idx = random.randint(0, total_vars - 1)
        x[idx] = random.choice(candidates)
        
    current_err = _compute_error(x[:U_size], x[U_size:U_size+V_size], x[U_size+V_size:], R, n, m, p, target_indices, zero_indices)
    
    best_x = x.copy()
    best_err = current_err
    
    T = 1.0
    T_min = 1e-10
    cooling_rate = 0.99
    
    steps = 0
    while time.time() - start_time < time_budget and T > T_min:
        steps += 1
        # Propose a move: change one random coefficient
        idx = random.randint(0, total_vars - 1)
        old_val = x[idx]
        new_val = random.choice(candidates)
        if new_val == old_val:
            continue
            
        x[idx] = new_val
        
        new_err = _compute_error(x[:U_size], x[U_size:U_size+V_size], x[U_size+V_size:], R, n, m, p, target_indices, zero_indices)
        
        delta_err = new_err - current_err
        
        if delta_err < 0 or random.random() < np.exp(-delta_err / T):
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_x = x.copy()
                
                # Check for convergence
                if best_err < 1e-8:
                    sol = _extract_and_verify(best_x[:U_size], best_x[U_size:U_size+V_size], best_x[U_size+V_size:])
                    if sol is not None:
                        return sol
        else:
            x[idx] = old_val
            
        T *= cooling_rate
        
        # If we get stuck at 0 error, we are done (or failed)
        if best_err < 1e-12:
             sol = _extract_and_verify(best_x[:U_size], best_x[U_size:U_size+V_size], best_x[U_size+V_size:])
             if sol is not None:
                 return sol
             # If it's very low but not verifiable as half-integer, try to polish with floats
             # but we can't return non-half-integers. So we break.
             break

    # Final check
    if best_err < 1e-8:
        sol = _extract_and_verify(best_x[:U_size], best_x[U_size:U_size+V_size], best_x[U_size+V_size:])
        if sol is not None:
            return sol

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Verify schoolbook
    if not _verify_rank(best["U"], best["V"], best["W"]):
        return best
    
    # Try to find a lower rank solution
    # The discrete search is slower per step than L-BFGS but might find exact solutions.
    # We try ranks in descending order.
    
    # Allocate time wisely.
    # Start from 59 down to 50.
    # Give more time to ranks closer to 60? Or spread it?
    # Let's try 59, 58, 57...
    
    rank_order = list(range(best_rank - 1, max(best_rank - 15, 0), -1))
    
    for R in rank_order:
        if time.time() > deadline - 10:
            break
            
        # Time budget per rank: 
        # If we have plenty of time, give 5s per rank.
        # If time is tight, give less.
        time_remaining = deadline - time.time() - 10
        ranks_remaining = len(rank_order) - rank_order.index(R)
        time_budget = max(2, min(8, time_remaining / max(1, ranks_remaining)))
        
        result = _try_discrete_search(R, n, m, p, time_budget=time_budget)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best

# EVOLVE-BLOCK-END

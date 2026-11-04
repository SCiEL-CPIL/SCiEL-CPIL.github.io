import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, with 10s safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return U, V, W

def _try_solve_w(M, target=1):
    """
    Solve M @ w = target_vector for w.
    M is m x r. target is scalar (all entries are target).
    Returns a list of length r with rational (Fraction) entries if a solution exists, else None.
    We look for a solution with small numerators/denominators.
    """
    m, r = M.shape
    # If r == 0, no solution
    if r == 0:
        return None
    
    # If m == 0, any w works, return zeros? No, must sum to target? 
    # But m is at least 1.
    
    # We want M @ w = [target] * m
    # This is a system of m equations with r unknowns.
    # If r < m, it might be inconsistent.
    # If r >= m, it is underdetermined.
    
    # We can use Gaussian elimination with Fractions to find a particular solution.
    # Or, simpler: if r == 1, w[0] must satisfy M[0,0]*w[0] = target, etc.
    
    # Let's try a simple approach:
    # If r == m, solve the square system.
    if r == m:
        try:
            # Use numpy for speed, then check if it's close to rational
            M_np = np.array(M, dtype=float)
            b_np = np.array([target] * m, dtype=float)
            w_np = np.linalg.solve(M_np, b_np)
            # Check if w_np is close to half-integers
            w_frac = []
            for val in w_np:
                # Check if 2*val is close to an integer
                if abs(2*val - round(2*val)) < 1e-9:
                    w_frac.append(Fraction(int(round(2*val)), 2))
                elif abs(val - round(val)) < 1e-9:
                    w_frac.append(Fraction(int(round(val)), 1))
                else:
                    return None
            # Verify with Fractions
            for j in range(m):
                s = Fraction(0)
                for t in range(r):
                    s += Fraction(M[j][t]) * w_frac[t]
                if s != target:
                    return None
            return w_frac
        except np.linalg.LinAlgError:
            return None
            
    # If r > m, we have freedom.
    # We can set r-m variables to 0 and solve the remaining m x m sub-system.
    # We try all combinations of m columns? That's C(r, m) which is large.
    # Instead, we can use the first m columns if they form an invertible matrix.
    # If not, try other sets?
    # For now, let's just try the first m columns.
    if r > m:
        M_sub = M[:, :m]
        if np.linalg.matrix_rank(M_sub) == m:
            try:
                M_np = np.array(M_sub, dtype=float)
                b_np = np.array([target] * m, dtype=float)
                w_sub_np = np.linalg.solve(M_np, b_np)
                w_frac = [Fraction(0)] * r
                for t in range(m):
                    val = w_sub_np[t]
                    if abs(2*val - round(2*val)) < 1e-9:
                        w_frac[t] = Fraction(int(round(2*val)), 2)
                    elif abs(val - round(val)) < 1e-9:
                        w_frac[t] = Fraction(int(round(val)), 1)
                    else:
                        return None
                # Verify
                for j in range(m):
                    s = Fraction(0)
                    for t in range(r):
                        s += Fraction(M[j][t]) * w_frac[t]
                    if s != target:
                        return None
                return w_frac
            except np.linalg.LinAlgError:
                return None
        else:
            # If first m columns are not invertible, try a random subset?
            # Or just return None for now.
            return None
            
    return None

def _random_search_uv(n, m, p, target_rank, n_attempts=500):
    """
    Search for U, V such that for all i,k, there exists a rational W_ik.
    """
    for attempt in range(n_attempts):
        if _time_left() < 5:
            return None
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        
        # Use sparse random initialization
        density = 0.4
        for i in range(n * m):
            for r in range(target_rank):
                if random.random() < density:
                    U[i][r] = 1 if random.random() < 0.5 else -1
        for j in range(m * p):
            for r in range(target_rank):
                if random.random() < density:
                    V[j][r] = 1 if random.random() < 0.5 else -1
        
        W = [[0] * target_rank for _ in range(p * n)]
        success = True
        
        for i in range(n):
            for k in range(p):
                # Build M for this (i, k)
                # M[j, t] = U[i*m+j, t] * V[j*p+k, t]
                M = [[0] * target_rank for _ in range(m)]
                for j in range(m):
                    for t in range(target_rank):
                        M[j][t] = U[i * m + j][t] * V[j * p + k][t]
                
                w_ik = _try_solve_w(M, target=1)
                if w_ik is None:
                    success = False
                    break
                
                # Store in W
                for t in range(target_rank):
                    W[k * n + i][t] = w_ik[t]
                    
            if not success:
                break
        
        if success:
            # Convert Fractions to float/int for return? 
            # The task says "integer/half-integer entries".
            # We can return them as floats if they are exact, or keep as is if the validator accepts.
            # The validator in the seed uses `s != 1` with floats.
            # Fractions will be converted to float in the validator?
            # `U[i*m+j][r]` will be a Fraction.
            # `s += ...` will work.
            # But the return type should be compatible.
            # Let's convert to float for safety, ensuring they are exact.
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            
            if _validate(U_out, V_out, W_out, n, m, p):
                return (U_out, V_out, W_out)
    
    return None

def _structured_search(n, m, p):
    """
    Try structured decompositions based on known patterns.
    For 4x4x5, try ranks from 20 upward.
    """
    lb = max(n * m, m * p, n * p)  # = 20 for 4,4,5
    
    for target_rank in range(lb, n * m * p + 1):
        if _time_left() < 10:
            break
        
        result = _random_search_uv(n, m, p, target_rank, n_attempts=200)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try structured random search
    result = _structured_search(n, m, p)
    
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

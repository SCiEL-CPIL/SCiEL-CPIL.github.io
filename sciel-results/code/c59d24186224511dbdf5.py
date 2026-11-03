import numpy as np
import time
import math
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110  # Safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _to_half_int_list(val):
    """Convert a value to a half-integer if possible, else return None."""
    # Check if val is a half-integer (multiple of 0.5)
    if isinstance(val, Fraction):
        # val = p/q, need q to divide 2
        if val.denominator == 1:
            return val.numerator
        elif val.denominator == 2:
            return val.numerator
        else:
            return None
    elif isinstance(val, float):
        # Check if close to a half-integer
        doubled = val * 2
        rounded = round(doubled)
        if abs(doubled - rounded) < 1e-9:
            return rounded
        return None
    elif isinstance(val, int):
        return val * 2
    return None

def _solve_for_W(U, V, n, m, p, r):
    """
    Given U (n*m x r) and V (m*p x r), solve for W (p*n x r)
    such that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    This is a system of linear equations in the W variables.
    """
    # Build the system: for each (i,k) pair, we have:
    # sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # But W[k*n+i,c] appears only in the equation for (i,k)
    # So for each (i,k), we need: sum_c M[i,k,c] * W[k*n+i,c] = 1
    # where M[i,k,c] = sum_j U[i*m+j,c]*V[j*p+k,c]
    
    # Actually, let me re-examine. The constraint is:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This means for each (i,k), the vector W[k*n+i,:] must satisfy:
    # For each j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # So for each (i,k), we have m equations in r unknowns (the components of W[k*n+i,:])
    # The equation for row j is: sum_c M_ikj_c * W[k*n+i,c] = 1
    # where M_ikj_c = U[i*m+j,c]*V[j*p+k,c]
    
    # We can try to find a sparse solution for each (i,k) independently.
    # For each (i,k), we have a system: A_ik * w_ik = 1_vector
    # where A_ik is an m x r matrix with A_ik[j,c] = U[i*m+j,c]*V[j*p+k,c]
    
    W = [[0]*r for _ in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            # Build the m x r matrix A_ik
            # A_ik[j,c] = U[i*m+j,c] * V[j*p+k,c]
            # We need to find w such that A_ik @ w = ones(m)
            
            # Try to find a sparse solution
            # First, check if there's a column that works for all j
            # i.e., a c such that U[i*m+j,c]*V[j*p+k,c] = 1 for all j
            found = False
            for c in range(r):
                works = True
                for j in range(m):
                    val = U[i*m+j][c] * V[j*p+k][c]
                    if val != 1:
                        works = False
                        break
                if works:
                    W[k*n+i][c] = 1
                    found = True
                    break
            
            if found:
                continue
            
            # Try two-term solutions
            # For each pair (c1, c2), see if we can find w1, w2 such that:
            # U[i*m+j,c1]*V[j*p+k,c1]*w1 + U[i*m+j,c2]*V[j*p+k,c2]*w2 = 1 for all j
            for c1 in range(r):
                for c2 in range(c1+1, r):
                    # For each j, we have:
                    # a_j * w1 + b_j * w2 = 1
                    # where a_j = U[i*m+j,c1]*V[j*p+k,c1], b_j = U[i*m+j,c2]*V[j*p+k,c2]
                    # If m >= 2, we can solve using the first two equations
                    if m >= 2:
                        a0 = U[i*m+0][c1] * V[j*p+k][c1] if False else U[i*m][c1] * V[0*p+k][c1]
                        b0 = U[i*m][c1] * V[0*p+k][c1]  # This is wrong, let me fix
                        # Let me redo this properly
                        a_vals = [U[i*m+j][c1] * V[j*p+k][c1] for j in range(m)]
                        b_vals = [U[i*m+j][c2] * V[j*p+k][c2] for j in range(m)]
                        
                        # Solve a0*w1 + b0*w2 = 1 and a1*w1 + b1*w2 = 1
                        a0, b0 = a_vals[0], b_vals[0]
                        if m > 1:
                            a1, b1 = a_vals[1], b_vals[1]
                        else:
                            a1, b1 = a0, b0
                        
                        # Determinant
                        det = a0*b1 - a1*b0
                        if det == 0:
                            continue
                        
                        w1 = (b1 - b0) / det  # This isn't right either
                        # Actually: a0*w1 + b0*w2 = 1, a1*w1 + b1*w2 = 1
                        # w1 = (b1 - b0) / (a0*b1 - a1*b0)? Let me use Cramer's rule properly
                        # w1 = (1*b1 - b0*1) / (a0*b1 - a1*b0) = (b1 - b0) / det
                        # w2 = (a0*1 - 1*a1) / (a0*b1 - a1*b0) = (a0 - a1) / det
                        
                        w1 = Fraction(b1 - b0, det)
                        w2 = Fraction(a0 - a1, det)
                        
                        # Check if this solution works for all j
                        valid = True
                        for j in range(m):
                            check = a_vals[j]*w1 + b_vals[j]*w2
                            if check != 1:
                                valid = False
                                break
                        
                        if valid:
                            # Check if w1, w2 are half-integers
                            h1 = _to_half_int_list(w1)
                            h2 = _to_half_int_list(w2)
                            if h1 is not None and h2 is not None:
                                W[k*n+i][c1] = h1
                                W[k*n+i][c2] = h2
                                found = True
                                break
                    else:
                        # m == 1, just need a0*w1 + b0*w2 = 1
                        # Try small half-integer values
                        for w1_half in range(-10, 11, 2):  # w1 = w1_half/2
                            w1 = Fraction(w1_half, 2)
                            if a0 == 0:
                                continue
                            w2 = (1 - a0*w1) / b0 if b0 != 0 else None
                            if w2 is None:
                                continue
                            h2 = _to_half_int_list(w2)
                            if h2 is not None:
                                W[k*n+i][c1] = w1_half
                                W[k*n+i][c2] = h2
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # Try single term: for some c, U[i*m+j,c]*V[j*p+k,c]*w = 1 for all j
                # This means U[i*m+j,c]*V[j*p+k,c] must be the same for all j, say val
                # Then w = 1/val
                for c in range(r):
                    vals = [U[i*m+j][c] * V[j*p+k][c] for j in range(m)]
                    if all(v == vals[0] for v in vals):
                        val = vals[0]
                        if val != 0:
                            w = Fraction(1, val)
                            hw = _to_half_int_list(w)
                            if hw is not None:
                                W[k*n+i][c] = hw
                                found = True
                                break
                if not found:
                    return None  # Can't solve for this (i,k)
    
    return W

def _try_random_pattern(n, m, p, r, seed):
    """Try a random binary pattern for U and V, then solve for W."""
    rng = random.Random(seed)
    
    # Create sparse random patterns
    # Each entry is -1, 0, or 1 with some probability of being nonzero
    density = 0.3
    
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    
    for i in range(n*m):
        for c in range(r):
            if rng.random() < density:
                U[i][c] = rng.choice([-1, 1])
    
    for j in range(m*p):
        for c in range(r):
            if rng.random() < density:
                V[j][c] = rng.choice([-1, 1])
    
    W = _solve_for_W(U, V, n, m, p, r)
    if W is None:
        return None
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_pattern(n, m, p, r, seed):
    """Try structured patterns based on Strassen-like ideas."""
    rng = random.Random(seed)
    
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    
    # For each column, assign a random subset of rows in U and V
    for c in range(r):
        # Choose which (i,j) entries are nonzero in column c of U
        num_u = rng.randint(1, min(3, n*m))
        u_indices = rng.sample(range(n*m), num_u)
        for idx in u_indices:
            U[idx][c] = rng.choice([-1, 1])
        
        # Choose which (j,k) entries are nonzero in column c of V
        num_v = rng.randint(1, min(3, m*p))
        v_indices = rng.sample(range(m*p), num_v)
        for idx in v_indices:
            V[idx][c] = rng.choice([-1, 1])
    
    W = _solve_for_W(U, V, n, m, p, r)
    if W is None:
        return None
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Try ranks from 12 up to best_rank - 1
    
    for target_rank in range(12, best_rank):
        if _time_remaining() < 5:
            break
        
        # Try multiple random seeds
        max_attempts = 50 if target_rank <= 20 else 30
        if _time_remaining() < 15:
            max_attempts = 10
        
        for attempt in range(max_attempts):
            if _time_remaining() < 2:
                break
            
            seed = attempt * 7919 + 12345
            
            # Try random pattern
            result = _try_random_pattern(n, m, p, target_rank, seed)
            if result is not None:
                best = result
                best_rank = target_rank
                break
            
            # Try structured pattern
            result = _try_structured_pattern(n, m, p, target_rank, seed + 1000)
            if result is not None:
                best = result
                best_rank = target_rank
                break
        
        # If we found a solution at this rank, continue to try lower
        # If not, continue to next rank
    
    return best

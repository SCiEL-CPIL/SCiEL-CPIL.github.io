import time
import numpy as np
import random

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_np(U, V, W, n, m, p):
    """Verify using numpy for speed."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j]
                v_val = V_arr[j*p+k]
                w_val = W_arr[k*n+i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(val, tol=1e-9):
    v = round(val * 2)
    return abs(val - v/2.0) < tol

def _to_half_integer(val):
    return round(val * 2) / 2.0

def _try_eliminate_terms(U, V, W, n, m, p, deadline, max_attempts=500):
    """Try to eliminate terms from the schoolbook decomposition by merging."""
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Candidate half-integer values for perturbations
    candidates = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2, 2.5, -2.5]
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Pick two random terms to try to merge
        c1 = random.randint(0, best_rank - 1)
        c2 = random.randint(0, best_rank - 1)
        if c1 == c2:
            continue
        
        # Try to eliminate term c2 by adding a perturbation to other terms
        # and adjusting c1 to compensate
        
        # Strategy: for each dimension (i,j,k), the contribution from term c is:
        # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want sum over c of this to equal 1.
        
        # Try: set U[i*m+j,c2] *= alpha for some pattern, and adjust c1
        # This is complex; instead try random perturbations and verify
        
        U_try = [row[:] for row in best_U]
        V_try = [row[:] for row in best_V]
        W_try = [row[:] for row in best_W]
        
        # Perturb a random subset of entries
        num_perturb = random.randint(1, 5)
        for _ in range(num_perturb):
            # Randomly modify an entry in one of the matrices
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n*m-1)
                col = random.randint(0, best_rank-1)
                U_try[row][col] = random.choice(candidates)
            elif mat == 'V':
                row = random.randint(0, m*p-1)
                col = random.randint(0, best_rank-1)
                V_try[row][col] = random.choice(candidates)
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, best_rank-1)
                W_try[row][col] = random.choice(candidates)
        
        # Try to zero out term c2 by setting its entries to 0
        # But this breaks the sum, so we need to compensate
        # Instead, just verify if the perturbed version works
        if _verify_np(U_try, V_try, W_try, n, m, p):
            # Check if we can remove any all-zero columns
            new_rank = 0
            new_U, new_V, new_W = [], [], []
            for c in range(best_rank):
                u_col = [U_try[r][c] for r in range(n*m)]
                v_col = [V_try[r][c] for r in range(m*p)]
                w_col = [W_try[r][c] for r in range(p*n)]
                if any(abs(x) > 1e-9 for x in u_col) or any(abs(x) > 1e-9 for x in v_col) or any(abs(x) > 1e-9 for x in w_col):
                    new_U.append([U_try[r][c] for r in range(n*m)])
                    new_V.append([V_try[r][c] for r in range(m*p)])
                    new_W.append([W_try[r][c] for r in range(p*n)])
                    new_rank += 1
            
            if new_rank < best_rank:
                # Transpose back to row-major format
                best_U = [[new_U[r][c] for c in range(new_rank)] for r in range(n*m)]
                best_V = [[new_V[r][c] for c in range(new_rank)] for r in range(m*p)]
                best_W = [[new_W[r][c] for c in range(new_rank)] for r in range(p*n)]
                best_rank = new_rank
    
    return best_U, best_V, best_W, best_rank

def _structured_search(n, m, p, deadline, target_rank):
    """Try to find a decomposition of exactly target_rank using structured patterns."""
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    # Use Kronecker product structure: 
    # For 3x4x6, we can try block structures
    # n=3, m=4, p=6
    
    # Try: use rank = n*p + m*p - p = p*(n+m-1) = 6*(3+4-1) = 36? No, that's not right.
    # Actually, for matrix multiplication, Strassen-like algorithms can reduce rank.
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # For our case, we can try to find patterns.
    
    # Simple approach: random search with half-integer constraints
    candidates = [0, 0.5, 1, -0.5, -1]
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        U = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(num_U)]
        V = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(num_V)]
        W = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(num_W)]
        
        if _verify_np(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook()
    best_U = [row[:] for row in schoolbook['U']]
    best_V = [row[:] for row in schoolbook['V']]
    best_W = [row[:] for row in schoolbook['W']]
    best_rank = n * m * p  # 72
    
    # Try to improve via elimination
    U, V, W, rank = _try_eliminate_terms(best_U, best_V, best_W, n, m, p, deadline, max_attempts=300)
    if rank < best_rank:
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try structured search for lower ranks
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But realistically much higher for rectangular matrix mult
    for r in range(best_rank - 1, 23, -1):
        if time.time() > deadline - 10:
            break
        result = _structured_search(n, m, p, deadline, r)
        if result is not None:
            U, V, W = result
            if _verify_np(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

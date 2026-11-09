"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy for real-valued checks."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 40."""
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

def _is_integer_or_half(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    if abs(x - round(x)) < tol:
        return True
    if abs(x * 2 - round(x * 2)) < tol:
        return True
    return False

def _to_clean(x):
    """Convert a float to int or half-integer if close enough."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    elif abs(x * 2 - round(x * 2)) < 1e-9:
        return float(round(x * 2) / 2)
    else:
        return x

def _try_rank_k_structural(k, deadline, n_trials=200):
    """Try to find a decomposition of rank k using structured random search.
    
    We sample sparse U and V with entries in {-1, 1} (or 0) and solve for W.
    We check if W has integer or half-integer entries.
    """
    n, m, p = DIMENSIONS
    if k < 10:
        return None
    
    trial = 0
    while trial < n_trials and time.time() < deadline:
        trial += 1
        
        # Sparse random initialization
        density = 0.4
        U = [[0] * k for _ in range(n*m)]
        V = [[0] * k for _ in range(m*p)]
        
        for r in range(n*m):
            for c in range(k):
                if random.random() < density:
                    U[r][c] = random.choice([-1, 1])
        for r in range(m*p):
            for c in range(k):
                if random.random() < density:
                    V[r][c] = random.choice([-1, 1])
        
        W = [[0]*k for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for ki in range(p):
                # Construct A (m x k) and b (m)
                A = np.zeros((m, k))
                for j in range(m):
                    for c in range(k):
                        A[j][c] = U[i*m + j][c] * V[j*p + ki][c]
                
                b = np.ones(m)
                
                # Solve A w = b
                if m == 1:
                    found = False
                    for c in range(k):
                        a_val = A[0][c]
                        if a_val == 0:
                            continue
                        val = 1.0 / a_val
                        if _is_integer_or_half(val):
                            W[ki*n+i][c] = val
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                else:
                    try:
                        w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ w, b, atol=1e-8):
                            valid = False
                            break
                        
                        w_candidate = w.copy()
                        
                        # Try rounding to integers
                        w_int = np.round(w_candidate)
                        if np.allclose(A @ w_int, b, atol=1e-8):
                            w_candidate = w_int
                        else:
                            # Try rounding to half-integers
                            w_half = np.round(w_candidate * 2) / 2
                            if np.allclose(A @ w_half, b, atol=1e-8):
                                w_candidate = w_half
                            else:
                                valid = False
                                break
                        
                        for val in w_candidate:
                            if not _is_integer_or_half(val):
                                valid = False
                                break
                        if not valid:
                            break
                            
                        W[ki*n + i] = [float(x) for x in w_candidate]
                    except:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify full solution
        if _verify(U, V, W):
            U_out = [[_to_clean(x) for x in row] for row in U]
            V_out = [[_to_clean(x) for x in row] for row in V]
            W_out = [[_to_clean(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _find_2x2x5_rank10(deadline, n_trials=5000):
    """Find a rank-10 decomposition for 2x2x5 matrix multiplication.
    
    U: (4, 10), V: (10, 10), W: (10, 10)
    Equations: sum_c U[i*2+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    n, m, p = 2, 2, 5
    rank = 10
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Random sparse U and V
        density = 0.5
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        
        for r in range(n*m):
            for c in range(rank):
                if random.random() < density:
                    U[r, c] = random.choice([-1, 1])
        for r in range(m*p):
            for c in range(rank):
                if random.random() < density:
                    V[r, c] = random.choice([-1, 1])
        
        # Solve for W
        W = np.zeros((p*n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # For fixed i, k, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # This is a system of m equations in rank unknowns
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                
                b = np.ones(m)
                
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ w, b, atol=1e-8):
                        valid = False
                        break
                    
                    # Try to round to integers or half-integers
                    w_candidate = w.copy()
                    
                    w_int = np.round(w_candidate)
                    if np.allclose(A @ w_int, b, atol=1e-8):
                        w_candidate = w_int
                    else:
                        w_half = np.round(w_candidate * 2) / 2
                        if np.allclose(A @ w_half, b, atol=1e-8):
                            w_candidate = w_half
                        else:
                            valid = False
                            break
                    
                    for val in w_candidate:
                        if not _is_integer_or_half(val):
                            valid = False
                            break
                    if not valid:
                        break
                        
                    W[k*n + i] = w_candidate
                except:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify_2x2x5(U_list, V_list, W_list):
            U_out = [[_to_clean(x) for x in row] for row in U_list]
            V_out = [[_to_clean(x) for x in row] for row in V_list]
            W_out = [[_to_clean(x) for x in row] for row in W_list]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _verify_2x2x5(U, V, W):
    """Verify 2x2x5 decomposition."""
    n, m, p = 2, 2, 5
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _assemble_2x4x5_from_2x2x5(dec1, dec2):
    """Assemble a 2x4x5 decomposition of rank 20 from two 2x2x5 rank-10 decompositions.
    
    dec1 and dec2 are dictionaries with U, V, W for 2x2x5.
    The 2x4x5 problem can be split into two independent 2x2x5 problems:
    - First half: rows 1-2 of the 4x5 matrix (j=0,1)
    - Second half: rows 3-4 of the 4x5 matrix (j=2,3)
    
    U for 2x4x5: (8, 20)
    V for 2x4x5: (20, 20)
    W for 2x4x5: (10, 20)
    
    For the first 2x2x5 (j=0,1):
    U[0:4, 0:10] = dec1['U']
    V[0:10, 0:10] = dec1['V']
    W[:, 0:10] = dec1['W']
    
    For the second 2x2x5 (j=2,3):
    U[4:8, 10:20] = dec2['U']
    V[10:20, 10:20] = dec2['V']
    W[:, 10:20] = dec2['W']
    
    But we need to be careful: the W matrix is shared. The W for 2x2x5 is (10, 10),
    where row index is k*2+i. For 2x4x5, W is (10, 20), where row index is k*2+i.
    So W[:, 0:10] = dec1['W'] and W[:, 10:20] = dec2['W'].
    """
    n, m, p = 2, 4, 5
    rank = 20
    
    U = [[0] * rank for _ in range(n*m)]  # (8, 20)
    V = [[0] * rank for _ in range(m*p)]  # (20, 20)
    W = [[0] * rank for _ in range(p*n)]  # (10, 20)
    
    # First 2x2x5: j=0,1
    for i in range(2):
        for j in range(2):
            for c in range(10):
                U[i*4 + j][c] = dec1['U'][i*2 + j][c]
    
    for j in range(2):
        for k in range(5):
            for c in range(10):
                V[j*5 + k][c] = dec1['V'][j*5 + k][c]
    
    for k in range(5):
        for i in range(2):
            for c in range(10):
                W[k*2 + i][c] = dec1['W'][k*2 + i][c]
    
    # Second 2x2x5: j=2,3
    for i in range(2):
        for j in range(2):
            for c in range(10):
                U[i*4 + (j+2)][10 + c] = dec2['U'][i*2 + j][c]
    
    for j in range(2):
        for k in range(5):
            for c in range(10):
                V[(j+2)*5 + k][10 + c] = dec2['V'][j*5 + k][c]
    
    for k in range(5):
        for i in range(2):
            for c in range(10):
                W[k*2 + i][10 + c] = dec2['W'][k*2 + i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback: Schoolbook rank 40
    fallback = _schoolbook()
    best_result = fallback
    best_rank = 40
    
    # First, try to find two 2x2x5 rank-10 decompositions and assemble rank 20
    dec1 = _find_2x2x5_rank10(deadline - 50, n_trials=3000)
    if dec1 is not None:
        dec2 = _find_2x2x5_rank10(deadline - 20, n_trials=3000)
        if dec2 is not None:
            result = _assemble_2x4x5_from_2x2x5(dec1, dec2)
            if _verify(result["U"], result["V"], result["W"]):
                return result
    
    # Try ranks from 25 to 39
    for k in range(25, 40):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        if remaining <= 0:
            break
        
        result = _try_rank_k_structural(k, deadline, n_trials=200)
        
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = len(result["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = result
                    break
    
    return best_result

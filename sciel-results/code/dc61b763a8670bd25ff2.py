import time
import sys
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the MM tensor.
    
    The MM tensor T has indices (i, j, k) with value 1 if and only if
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k).
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape for efficient tensor contraction
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W index is k*n+i, so W[k,n,i] in 3D (p,n,rank) is W_reshaped[k, i, c]
    
    # Efficient computation using einsum
    # T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note the indices:
    # U: i, j, c
    # V: j, k, c
    # W: k, i, c (since W index is k*n+i, row k, col i in the p x n grid)
    
    # Let's verify the index mapping for W.
    # W is (p*n, rank). Row index is k*n + i.
    # So W[k*n+i, c] is the element.
    # In the reshaped W_reshaped (p, n, rank), W_reshaped[k, i, c] corresponds to W[k*n+i, c].
    
    # We want sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Use einsum:
    # 'ijc,jkc,kic->ijk'
    # But we need to be careful with memory. n=3, m=4, p=8, rank~96.
    # 3*4*8*96 is small.
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T is all ones
    # The tensor for matrix multiplication has 1s at all (i,j,k) positions?
    # No, the matrix multiplication tensor T_{ijk} is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor decomposition defines the bilinear map.
    # The "tensor" usually referred to is the structure tensor of the algebra.
    # For matrix multiplication, the tensor T has entries T_{ijk} = 1 if the term A_{ij}B_{jk} appears in C_{ik}.
    # Wait, the standard definition:
    # The multiplication tensor T for m_{n,p} has indices (i, j, k) where i in [n], j in [m], k in [p].
    # T_{ijk} = 1.
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # u_c is in R^{n*m} (indexed by i,j), v_c is in R^{m*p} (indexed by j,k), w_c is in R^{p*n} (indexed by k,i).
    # The condition is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    # So yes, we check if T is the all-ones tensor of shape (n, m, p).
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is integer)."""
    return abs(2.0 * x - round(2.0 * x)) < 1e-9

def _all_half_integers(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _clean(mat):
    """Clean entries to nearest integer or half-integer."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            # Snap to nearest 0.5
            snapped = round(val * 2.0) / 2.0
            if abs(snapped - round(snapped)) < 1e-9:
                new_row.append(int(round(snapped)))
            else:
                new_row.append(snapped)
        result.append(new_row)
    return result

def _schoolbook(n, m, p):
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

def _objective(params, n, m, p, rank):
    """Objective function for optimization: sum of squared errors of tensor reconstruction."""
    # Unpack params into U, V, W
    # U size: n*m*rank
    # V size: m*p*rank
    # W size: p*n*rank
    
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : ]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Target is all ones
    error = T - 1.0
    return np.sum(error * error)

def _try_relax_and_round(n, m, p, target_rank, init_U, init_V, init_W, time_limit=10.0):
    """
    Attempt to find a valid decomposition of target_rank by optimizing
    a continuous relaxation of the problem, starting from an initial guess.
    """
    start_time = time.time()
    
    # Flatten initial guess
    U_init = np.array(init_U, dtype=np.float64)
    V_init = np.array(init_V, dtype=np.float64)
    W_init = np.array(init_W, dtype=np.float64)
    
    # If init_U is larger than target_rank, we need to truncate or project.
    # Here we assume init_U is already of size (n*m, target_rank) or we handle it.
    # For simplicity, this function expects init matrices of the correct target_rank size.
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use L-BFGS-B or Nelder-Mead
    # Nelder-Mead is derivative-free and might be more robust for this non-convex problem,
    # but slower. L-BFGS-B requires gradients. Let's use L-BFGS-B with numerical gradients
    # or just Nelder-Mead if dimensions are small.
    # Dim: (12+32+24)*rank. For rank 95, ~100*95 = 9500 variables.
    # Nelder-Mead on 9500 variables is very slow.
    # L-BFGS-B is better.
    
    # We will use L-BFGS-B.
    # To speed up, we might reduce the number of iterations.
    
    options = {
        'maxiter': 1000,
        'ftol': 1e-10,
        'gtol': 1e-8,
        'disp': False
    }
    
    try:
        res = minimize(_objective, x0, args=(n, m, p, target_rank), method='L-BFGS-B', options=options)
        
        if time.time() - start_time > time_limit:
            return None
            
        # Check if the continuous solution is close to integer/half-integer
        # and if the error is small.
        if res.fun < 1e-6:
            # Extract and clean
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            U_clean = _clean(U_opt.tolist())
            V_clean = _clean(V_opt.tolist())
            W_clean = _clean(W_opt.tolist())
            
            if _verify(U_clean, V_clean, W_clean, n, m, p):
                return {"U": U_clean, "V": V_clean, "W": W_clean}
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Verify schoolbook (should always pass)
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to pure integers if verification fails (shouldn't happen)
        return best
    
    # Try to reduce rank
    # We will attempt to find a solution of rank R-1, R-2, etc.
    # Strategy: 
    # 1. Start with schoolbook.
    # 2. To try rank R-1, we can't just drop a column. We need to find a new set of R-1 columns.
    # 3. We can initialize the search for rank R-1 by taking the schoolbook solution,
    #    removing one column, and adding a small random perturbation to the remaining columns,
    #    then using the optimizer to try to close the gap.
    
    # However, a more effective heuristic for "relax and round" is to start with a random
    # initialization of the target rank and hope the optimizer finds a solution.
    # Given the high dimension, this is unlikely.
    
    # Alternative: Try to merge two schoolbook terms.
    # If we can express two basis tensors as a sum of products of lower rank terms...
    
    # Let's try a few random restarts of the "Relax and Round" approach for rank best_rank - 1.
    # We will use the schoolbook solution as a "hint" by initializing the first best_rank-1
    # columns with the schoolbook values and the last column with random noise, then optimizing.
    # Actually, just random initialization is better for exploring the landscape.
    
    max_attempts_per_rank = 5
    
    for rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 20.0:
            break
        
        found = False
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize with random half-integers
            rng = random.Random(attempt * 1000 + rank)
            U_init = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0]) for _ in range(rank)] for _ in range(n * m)]
            V_init = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0]) for _ in range(rank)] for _ in range(m * p)]
            W_init = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0]) for _ in range(rank)] for _ in range(p * n)]
            
            # Also try initializing with schoolbook columns (truncated)
            if attempt == 0:
                # Use schoolbook columns for the first 'rank' terms
                U_init = [row[:rank] for row in best["U"]]
                V_init = [row[:rank] for row in best["V"]]
                W_init = [row[:rank] for row in best["W"]]
                # Add small noise to avoid local minima of the truncated solution
                for i in range(n * m):
                    for c in range(rank):
                        U_init[i][c] += rng.uniform(-0.1, 0.1)
                for i in range(m * p):
                    for c in range(rank):
                        V_init[i][c] += rng.uniform(-0.1, 0.1)
                for i in range(p * n):
                    for c in range(rank):
                        W_init[i][c] += rng.uniform(-0.1, 0.1)
            
            result = _try_relax_and_round(n, m, p, rank, U_init, V_init, W_init, time_limit=15.0)
            
            if result:
                best = result
                best_rank = rank
                found = True
                break
        
        if found:
            # Found a better one, continue searching for even lower
            continue
    
    # Ensure entries are proper numeric types
    best["U"] = _clean(best["U"])
    best["V"] = _clean(best["V"])
    best["W"] = _clean(best["W"])
    
    return best
# EVOLVE-BLOCK-END

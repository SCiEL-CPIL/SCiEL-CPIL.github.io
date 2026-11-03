import numpy as np
import time
import sys
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Pre-compute indices for fast error calculation
    # We need to map (i, j, k) to indices in U, V, W
    # U[i, j] -> index i*m + j
    # V[j, k] -> index j*p + k
    # W[k, i] -> index k*n + i
    
    # We will store U, V, W as 2D arrays (rows, cols) where cols are rank terms
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    def calc_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Residual for (i, j, k) is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        # We sum squared residuals
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    # Sum over rank
                    # Vectorized dot product of 3 vectors
                    # But we are in a loop, so we can just loop or use np.dot
                    # np.dot is faster for small vectors?
                    # Let's use manual loop for clarity or np.einsum if we pre-extract rows
                    # Actually, extracting rows every time is expensive.
                    # Let's just loop over c. R is small (~30).
                    for c in range(U.shape[1]):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    diff = s - 1.0
                    err += diff * diff
        return err

    def local_search(R, max_steps, initial_U=None, initial_V=None, initial_W=None):
        if initial_U is None:
            # Initialize with sparse random half-integers
            # Most zeros, some 0.5, -0.5, 1, -1
            U = np.zeros((size_U, R))
            V = np.zeros((size_V, R))
            W = np.zeros((size_W, R))
            
            # Add some structure: ensure each column has some non-zeros
            # Randomly assign values
            values = [0.5, -0.5, 1.0, -1.0, 0.0]
            for c in range(R):
                # Pick a few random positions to be non-zero
                num_nonzero_U = random.randint(2, 5)
                num_nonzero_V = random.randint(2, 5)
                num_nonzero_W = random.randint(2, 5)
                
                rows_U = random.sample(range(size_U), num_nonzero_U)
                for r in rows_U:
                    U[r, c] = random.choice(values[:4]) # Exclude 0 for non-zero slots
                    
                rows_V = random.sample(range(size_V), num_nonzero_V)
                for r in rows_V:
                    V[r, c] = random.choice(values[:4])
                    
                rows_W = random.sample(range(size_W), num_nonzero_W)
                for r in rows_W:
                    W[r, c] = random.choice(values[:4])
        else:
            U = initial_U.copy()
            V = initial_V.copy()
            W = initial_W.copy()
            
        current_err = calc_error(U, V, W)
        
        # If already 0, return
        if current_err < 1e-9:
            return U, V, W, 0.0
            
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random matrix and element
            mat_choice = random.randint(0, 2)
            
            if mat_choice == 0:
                mat = U
                rows = size_U
            elif mat_choice == 1:
                mat = V
                rows = size_V
            else:
                mat = W
                rows = size_W
                
            r = random.randint(0, rows - 1)
            c = random.randint(0, R - 1)
            
            old_val = mat[r, c]
            
            # Try changing to old_val + 0.5 or old_val - 0.5
            # Or maybe jump to a random nearby value
            candidates = []
            new_val = old_val + 0.5
            if abs(new_val) <= 2.0:
                candidates.append(new_val)
            new_val = old_val - 0.5
            if abs(new_val) <= 2.0:
                candidates.append(new_val)
                
            # Also try 0 if not already 0
            if old_val != 0.0:
                candidates.append(0.0)
                
            # If old_val is 0, try 0.5 and -0.5 (already covered)
            
            best_new_val = old_val
            best_new_err = current_err
            
            for val in candidates:
                if val == old_val:
                    continue
                mat[r, c] = val
                new_err = calc_error(U, V, W)
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_new_val = val
                mat[r, c] = old_val # Restore for next candidate
                
            if best_new_val != old_val:
                mat[r, c] = best_new_val
                current_err = best_new_err
                
                if current_err < 1e-9:
                    return U, V, W, current_err
                    
        return U, V, W, current_err

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, n*p) = 20
    # But due to structure, it might be higher.
    # We'll try 25, 30, 35, 40, 45, 50, 55, 60
    
    rank_list = [25, 30, 35, 40, 45, 50, 55, 60]
    
    for R in rank_list:
        if time.time() > deadline - 20:
            break
            
        # Try a few restarts for this rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Adjust max_steps based on remaining time?
            # Let's fix it to 2000 steps per restart.
            max_steps = 2000
            
            U, V, W, err = local_search(R, max_steps)
            
            if err < 1e-9:
                # Found a solution!
                # Ensure it's half-integer (it is, by construction)
                # Convert to lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback: Schoolbook decomposition
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Pre-compute indices for fast error calculation
    # We need to map (i, j, k) to indices in U, V, W
    # U[i, j] -> index i*m + j
    # V[j, k] -> index j*p + k
    # W[k, i] -> index k*n + i
    
    def calc_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Residual for (i, j, k) is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        # We sum squared residuals
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    # Sum over rank
                    # Vectorized dot product of 3 vectors
                    # But we are in a loop, so we can just loop or use np.dot
                    # Let's use manual loop for clarity or np.einsum if we pre-extract rows
                    # Actually, extracting rows every time is expensive.
                    # Let's just loop over c. R is small (~30).
                    for c in range(U.shape[1]):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    diff = s - 1.0
                    err += diff * diff
        return err

    def local_search(R, max_steps, initial_U=None, initial_V=None, initial_W=None):
        if initial_U is None:
            # Initialize with sparse random half-integers
            # Most zeros, some 0.5, -0.5, 1, -1
            U = np.zeros((size_U, R))
            V = np.zeros((size_V, R))
            W = np.zeros((size_W, R))
            
            # Add some structure: ensure each column has some non-zeros
            # Randomly assign values
            values = [0.5, -0.5, 1.0, -1.0, 0.0]
            for c in range(R):
                # Pick a few random positions to be non-zero
                num_nonzero_U = random.randint(2, 5)
                num_nonzero_V = random.randint(2, 5)
                num_nonzero_W = random.randint(2, 5)
                
                rows_U = random.sample(range(size_U), num_nonzero_U)
                for r in rows_U:
                    U[r, c] = random.choice(values[:4]) # Exclude 0 for non-zero slots
                    
                rows_V = random.sample(range(size_V), num_nonzero_V)
                for r in rows_V:
                    V[r, c] = random.choice(values[:4])
                    
                rows_W = random.sample(range(size_W), num_nonzero_W)
                for r in rows_W:
                    W[r, c] = random.choice(values[:4])
        else:
            U = initial_U.copy()
            V = initial_V.copy()
            W = initial_W.copy()
            
        current_err = calc_error(U, V, W)
        
        # If already 0, return
        if current_err < 1e-9:
            return U, V, W, 0.0
            
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random matrix and element
            mat_choice = random.randint(0, 2)
            
            if mat_choice == 0:
                mat = U
                rows = size_U
            elif mat_choice == 1:
                mat = V
                rows = size_V
            else:
                mat = W
                rows = size_W
                
            r = random.randint(0, rows - 1)
            c = random.randint(0, R - 1)
            
            old_val = mat[r, c]
            
            # Try changing to old_val + 0.5 or old_val - 0.5
            # Or maybe jump to a random nearby value
            candidates = []
            new_val = old_val + 0.5
            if abs(new_val) <= 2.0:
                candidates.append(new_val)
            new_val = old_val - 0.5
            if abs(new_val) <= 2.0:
                candidates.append(new_val)
                
            # Also try 0 if not already 0
            if old_val != 0.0:
                candidates.append(0.0)
                
            # If old_val is 0, try 0.5 and -0.5 (already covered)
            
            best_new_val = old_val
            best_new_err = current_err
            
            for val in candidates:
                if val == old_val:
                    continue
                mat[r, c] = val
                new_err = calc_error(U, V, W)
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_new_val = val
                mat[r, c] = old_val # Restore for next candidate
                
            if best_new_val != old_val:
                mat[r, c] = best_new_val
                current_err = best_new_err
                
                if current_err < 1e-9:
                    return U, V, W, current_err
                    
        return U, V, W, current_err

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, n*p) = 20
    # But due to structure, it might be higher.
    # We'll try 25, 30, 35, 40, 45, 50, 55, 60
    
    rank_list = [25, 30, 35, 40, 45, 50, 55, 60]
    
    for R in rank_list:
        if time.time() > deadline - 20:
            break
            
        # Try a few restarts for this rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Adjust max_steps based on remaining time?
            # Let's fix it to 2000 steps per restart.
            max_steps = 2000
            
            U, V, W, err = local_search(R, max_steps)
            
            if err < 1e-9:
                # Found a solution!
                # Ensure it's half-integer (it is, by construction)
                # Convert to lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback: Schoolbook decomposition
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
# EVOLVE-BLOCK-END

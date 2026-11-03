"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = 2, 3, 4
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def calculate_error(U, V, W, r):
        """Calculate the squared error of the decomposition."""
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r_idx in range(r):
                        total += U[i*m + j][r_idx] * V[j*p + k][r_idx] * W[k*n + i][r_idx]
                    error += (total - 1) ** 2
        return error

    def local_search(U, V, W, r, time_limit):
        """Perform local search to improve the decomposition."""
        current_error = calculate_error(U, V, W, r)
        if current_error == 0:
            return True, U, W
        
        # Pre-compute indices
        u_indices = [(i*m + j) for i in range(n) for j in range(m)]
        v_indices = [(j*p + k) for j in range(m) for k in range(p)]
        w_indices = [(k*n + i) for k in range(p) for i in range(n)]
        
        all_vars = []
        for idx in u_indices:
            for r_idx in range(r):
                all_vars.append(('U', idx, r_idx))
        for idx in v_indices:
            for r_idx in range(r):
                all_vars.append(('V', idx, r_idx))
        for idx in w_indices:
            for r_idx in range(r):
                all_vars.append(('W', idx, r_idx))
        
        while time.time() < time_limit:
            # Pick a random variable
            var_type, idx1, idx2 = random.choice(all_vars)
            
            # Get old value
            if var_type == 'U':
                old_val = U[idx1][idx2]
            elif var_type == 'V':
                old_val = V[idx1][idx2]
            else:
                old_val = W[idx1][idx2]
                
            # Try new values
            for delta in [-1, 1, -2, 2]:
                new_val = old_val + delta
                if new_val < -10 or new_val > 10:
                    continue
                
                # Apply change
                if var_type == 'U':
                    U[idx1][idx2] = new_val
                elif var_type == 'V':
                    V[idx1][idx2] = new_val
                else:
                    W[idx1][idx2] = new_val
                
                new_error = calculate_error(U, V, W, r)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        return True, U, W
                else:
                    # Revert
                    if var_type == 'U':
                        U[idx1][idx2] = old_val
                    elif var_type == 'V':
                        V[idx1][idx2] = old_val
                    else:
                        W[idx1][idx2] = old_val
            
            # Occasionally, do a larger jump to escape local minima
            if random.random() < 0.05:
                new_val = random.randint(-10, 10)
                if var_type == 'U':
                    U[idx1][idx2] = new_val
                elif var_type == 'V':
                    V[idx1][idx2] = new_val
                else:
                    W[idx1][idx2] = new_val
                
                new_error = calculate_error(U, V, W, r)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        return True, U, W
                else:
                    # Revert
                    if var_type == 'U':
                        U[idx1][idx2] = old_val
                    elif var_type == 'V':
                        V[idx1][idx2] = old_val
                    else:
                        W[idx1][idx2] = old_val
        
        return False, U, W

    def search_for_rank(r, end_time):
        """Try to find a rank-r decomposition before end_time."""
        while time.time() < end_time:
            U = [[random.randint(-5, 5) for _ in range(r)] for _ in range(n*m)]
            V = [[random.randint(-5, 5) for _ in range(r)] for _ in range(m*p)]
            W = [[random.randint(-5, 5) for _ in range(r)] for _ in range(p*n)]
            
            success, U, W = local_search(U, V, W, r, end_time)
            if success:
                return U, V, W
        return None

    # Allocate 30s to rank 20, 30s to rank 21, 30s to rank 22
    result = search_for_rank(20, start_time + 30)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    result = search_for_rank(21, start_time + 60)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    result = search_for_rank(22, start_time + 90)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
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

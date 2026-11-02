import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def compute_residuals(U, V, R):
        """
        Computes the sum of squared residuals for the systems M_{i,k} w = 1.
        Returns total_residual, residuals_per_pair.
        """
        total_res = 0.0
        res_per_pair = {}
        for i in range(n):
            for k in range(p):
                # Construct M (m x R)
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i*m + j][r] * V[j*p + k][r]
                
                # Solve M w = 1
                try:
                    w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    if residuals is not None:
                        res = residuals[0]
                    else:
                        # Compute residual manually if lstsq doesn't return it
                        res = np.sum((M @ w - np.ones(m))**2)
                except:
                    res = 1e10
                    
                res_per_pair[(i, k)] = res
                total_res += res
        return total_res, res_per_pair

    def snap_and_verify(U, V, R):
        """
        Snaps U, V to nearest 0.5 and checks if W can be solved exactly.
        Returns dict if valid, else None.
        """
        U_snap = [[round(u * 2) / 2 for u in row] for row in U]
        V_snap = [[round(v * 2) / 2 for v in row] for row in V]
        
        W = [[0] * R for _ in range(pn)]
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U_snap[i*m + j][r] * V_snap[j*p + k][r]
                
                try:
                    w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    if residuals is not None and residuals[0] > 1e-9:
                        return None
                    # Check if w is close to 0.5 increments
                    w_snap = [round(x * 2) / 2 for x in w]
                    # Re-check with snapped W
                    M_w = M @ np.array(w_snap)
                    if not np.allclose(M_w, np.ones(m), atol=1e-9):
                        return None
                    for r in range(R):
                        W[k*n + i][r] = w_snap[r]
                except:
                    return None
        
        # Final verification
        if verify_decomposition(U_snap, V_snap, W, R):
            return {"U": U_snap, "V": V_snap, "W": W}
        return None

    def simulated_annealing(R, time_limit):
        """
        Runs SA to find U, V such that residuals are near 0.
        """
        # Initialize with small integers
        U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
        
        current_res, _ = compute_residuals(U, V, R)
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_res = current_res
        
        T = 1.0
        cooling = 0.999
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        while time.time() < time_limit:
            # Propose move
            if random.random() < 0.5:
                # Change U
                i = random.randint(0, nm - 1)
                r = random.randint(0, R - 1)
                old_val = U[i][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                U[i][r] = new_val
                
                # Affected pairs: (i//m, k) for all k
                # i corresponds to (i_row, j_row) where i_row = i//m
                i_row = i // m
                affected = [(i_row, k) for k in range(p)]
            else:
                # Change V
                j = random.randint(0, mp - 1)
                r = random.randint(0, R - 1)
                old_val = V[j][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                V[j][r] = new_val
                
                # j corresponds to (j_row, k_row) where j_row = j//p
                k_row = j % p
                # Affected pairs: (i, k_row) for all i
                affected = [(i, k_row) for i in range(n)]
            
            # Compute new residual (approximate by recomputing affected pairs)
            # For simplicity and correctness, recompute all if R is small, else affected
            # Given R is small (<=56), recomputing all is fast enough (14 pairs * 4x56 matrix)
            new_res, _ = compute_residuals(U, V, R)
            
            delta = new_res - current_res
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_res = new_res
                if new_res < best_res:
                    best_res = new_res
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    
                    # If very good, try to snap and verify
                    if best_res < 1e-6:
                        result = snap_and_verify(best_U, best_V, R)
                        if result:
                            return result
            else:
                # Revert
                if random.random() < 0.5: # This logic is flawed, need to track which was changed
                    pass # The revert logic above is missing. Let's fix.
            
            T *= cooling
            if T < 1e-4:
                T = 1e-4
        
        # Final attempt with best found
        if best_res < 1e-6:
            result = snap_and_verify(best_U, best_V, R)
            if result:
                return result
        return None

    # Fix SA revert logic
    def simulated_annealing_v2(R, time_limit):
        U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
        
        current_res, _ = compute_residuals(U, V, R)
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_res = current_res
        
        T = 1.0
        cooling = 0.9995
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        while time.time() < time_limit:
            if random.random() < 0.5:
                i = random.randint(0, nm - 1)
                r = random.randint(0, R - 1)
                old_val = U[i][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                U[i][r] = new_val
                changed = ('U', i, r, old_val)
            else:
                j = random.randint(0, mp - 1)
                r = random.randint(0, R - 1)
                old_val = V[j][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                V[j][r] = new_val
                changed = ('V', j, r, old_val)
            
            new_res, _ = compute_residuals(U, V, R)
            delta = new_res - current_res
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_res = new_res
                if new_res < best_res:
                    best_res = new_res
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    
                    if best_res < 1e-6:
                        result = snap_and_verify(best_U, best_V, R)
                        if result:
                            return result
            else:
                # Revert
                if changed[0] == 'U':
                    U[changed[1]][changed[2]] = changed[3]
                else:
                    V[changed[1]][changed[2]] = changed[3]
            
            T *= cooling
            if T < 1e-4:
                T = 1e-4
        
        if best_res < 1e-6:
            result = snap_and_verify(best_U, best_V, R)
            if result:
                return result
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    min_possible_rank = max(nm, mp, pn)
    
    # Try ranks from lower bound up to schoolbook
    # Allocate time based on rank difficulty
    # Lower ranks are harder, give them more time? Or just iterate.
    
    for R in range(min_possible_rank, best_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Give each rank a chunk of time
        # If we have 100s left, and 28 ranks to try, ~3s per rank?
        # Better: try a few ranks with longer time.
        
        time_limit = time.time() + 5.0
        if R < 35:
            time_limit = time.time() + 10.0
            
        result = simulated_annealing_v2(R, time_limit)
        if result is not None:
            best_result = result
            best_rank = R
            break
            
    return best_result

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def polish(U, V, W, n, m, p, max_steps=50):
        """
        Try to adjust V and W slightly to make the decomposition exact.
        U is fixed to half-integers.
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        # Convert to float for numpy operations
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        def error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                        err += (val - 1.0)**2
            return err
        
        current_err = error()
        if current_err < 1e-10:
            return True
            
        for _ in range(max_steps):
            if time.time() > deadline:
                return False
                
            # Pick a random variable to flip/adjust
            # We adjust V or W entries.
            # Let's try flipping signs or adding 1.
            # Since we want half-integers, let's stick to integers for V/W in this polish step?
            # Or maybe V/W are already integers?
            # Let's assume V and W are integers and we try to change one entry by +/- 1.
            
            # Randomly select a matrix and an index
            mat_choice = random.choice(['V', 'W'])
            if mat_choice == 'V':
                idx = random.randrange(mp)
                col = random.randrange(rank)
                old_val = V_np[idx, col]
                # Try +1, -1, or flip sign?
                # Let's try +1 and -1
                deltas = [1, -1]
                random.shuffle(deltas)
                improved = False
                for d in deltas:
                    V_np[idx, col] += d
                    new_err = error()
                    if new_err < current_err:
                        current_err = new_err
                        V[idx][col] += d
                        improved = True
                        break
                    else:
                        V_np[idx, col] -= d # revert
                if not improved:
                    # Try changing U? No, U is fixed to half-integers.
                    # If V/W polish fails, we might need to change the structure.
                    # For now, break if no improvement found in a few steps?
                    # Let's just continue to next step
                    pass
            else:
                idx = random.randrange(pn)
                col = random.randrange(rank)
                old_val = W_np[idx, col]
                deltas = [1, -1]
                random.shuffle(deltas)
                improved = False
                for d in deltas:
                    W_np[idx, col] += d
                    new_err = error()
                    if new_err < current_err:
                        current_err = new_err
                        W[idx][col] += d
                        improved = True
                        break
                    else:
                        W_np[idx, col] -= d
                if not improved:
                    pass
            
            if current_err < 1e-10:
                return True
                
        return False

    def try_rank(r, num_attempts=100, seed_base=0):
        """
        Try to find a decomposition of rank r.
        Strategy: random V, W; solve for U; round to half-integers; verify/polish.
        """
        if time.time() > deadline:
            return None
        
        candidates = []
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            random.seed(seed_base + r * 1000 + attempt)
            
            # Generate random V and W with small integer entries
            # Use a mix of -1, 0, 1, 2 to allow more structure
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(pn)]
            
            U = [[0.0]*r for _ in range(nm)]
            valid = True
            max_diff = 0.0
            
            for i in range(n):
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Round to nearest half-integer
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))
                if diff > max_diff:
                    max_diff = diff
                
                # Also check integer rounding
                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))
                if diff_int < diff:
                    sol_rounded = sol_int
                    diff = diff_int
                    if diff > max_diff:
                        max_diff = diff
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # If the max difference is small, it's a good candidate
            if max_diff < 0.25: # Closer than 0.25 means it's likely 0 or 0.5
                candidates.append((max_diff, U, V, W))
        
        # Sort candidates by error
        candidates.sort(key=lambda x: x[0])
        
        # Try to verify or polish the best candidates
        for diff, U, V, W in candidates[:10]: # Top 10
            if time.time() > deadline:
                break
                
            # First, try exact verification with the rounded values
            if _verify_np(U, V, W, n, m, p):
                U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
                
                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
            
            # If direct verification fails, try polishing V and W
            # Note: U is fixed to half-integers.
            # We need to ensure V and W remain integers? The problem says "integer/half-integer entries".
            # So V and W can also be half-integers.
            # My polish function above assumes V/W are integers and adds integers.
            # Let's assume V/W are integers for now, as they are generated as such.
            
            if polish(U, V, W, n, m, p, max_steps=100):
                if _verify_np(U, V, W, n, m, p):
                    U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                    V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                    W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
                    
                    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                        return {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }
        
        return None

    best_result = None
    best_rank = n * m * p
    
    # Lower bound is max(nm, mp, pn) = 35.
    # Search from 35 upwards.
    
    # Ranks 35 to 50: 16 ranks * 200 attempts = 3200 attempts
    # Ranks 51 to 70: 20 ranks * 100 attempts = 2000 attempts
    # Ranks 71 to 104: 34 ranks * 50 attempts = 1700 attempts
    
    for r in range(35, 51):
        if time.time() > deadline:
            break
        result = try_rank(r, num_attempts=200, seed_base=0)
        if result is not None:
            best_result = result
            best_rank = r
            break
            
    if best_result is None:
        for r in range(51, 71):
            if time.time() > deadline:
                break
            result = try_rank(r, num_attempts=100, seed_base=0)
            if result is not None:
                best_result = result
                best_rank = r
                break
                
    if best_result is None:
        for r in range(71, 105):
            if time.time() > deadline:
                break
            result = try_rank(r, num_attempts=50, seed_base=0)
            if result is not None:
                best_result = result
                best_rank = r
                break
            
    if best_result is not None:
        return best_result
    
    return schoolbook()

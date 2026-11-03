import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 140  # seconds, safety margin before 150s kill
    
    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if val != 1:
                        return False
        return True

    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_list(x) for x in obj]
        else:
            return obj

    # Strategy: Decompose T_3,3,4 into T_3,3,3 (k=0,1,2) and T_3,3,1 (k=3)
    # T_3,3,1 has rank 9.
    # T_3,3,3 has rank <= 23.
    # Total rank <= 32.
    
    # Pre-define the 9 ranks for the k=3 part
    # U: (9, 32), V: (12, 32), W: (12, 32)
    # We will search for the first 23 ranks for the k=0,1,2 part.
    # The last 9 ranks are fixed for k=3.
    
    def find_rank_3x3x3(R_target, max_attempts=5000):
        """
        Find a rank R decomposition for 3x3 x 3x3 matrix multiplication.
        Returns U, V, W as numpy arrays of shape (9, R), (9, R), (9, R) if found, else None.
        """
        n3, m3, p3 = 3, 3, 3
        # Indices for 3x3x3:
        # U[i*m3+j, r]
        # V[j*p3+k, r]
        # W[k*n3+i, r]
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit - 10:
                return None
            
            # Random integers
            # Use a range that is likely to contain solutions
            # Small integers are preferred
            U_np = np.random.randint(-2, 3, size=(n3 * m3, R_target))
            V_np = np.random.randint(-2, 3, size=(m3 * p3, R_target))
            
            # Build linear system for W
            # Variables: W[k*n3+i, r] for k in 0..2, i in 0..2, r in 0..R_target-1
            # Total vars: 9 * R_target
            # Equations: 3*3*3 = 27
            num_vars = p3 * n3 * R_target
            A_mat = np.zeros((n3 * m3 * p3, num_vars))
            b_vec = np.ones(n3 * m3 * p3)
            
            eq_idx = 0
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        u_idx = i * m3 + j
                        v_idx = j * p3 + k
                        w_base = k * n3 + i
                        for r in range(R_target):
                            var_idx = w_base * R_target + r
                            A_mat[eq_idx, var_idx] = U_np[u_idx, r] * V_np[v_idx, r]
                        eq_idx += 1
            
            try:
                # Solve for W
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check for integer solution
                W_int = np.round(W_flat)
                if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                    W_sol = W_int.reshape((p3 * n3, R_target))
                    # Verify
                    # Convert to Fraction for exact check
                    U_list = [[Fraction(x) for x in row] for row in U_np.tolist()]
                    V_list = [[Fraction(x) for x in row] for row in V_np.tolist()]
                    W_list = [[Fraction(x) for x in row] for row in W_sol.tolist()]
                    
                    # Local validity check for 3x3x3
                    valid = True
                    for i in range(n3):
                        for j in range(m3):
                            for k in range(p3):
                                val = Fraction(0)
                                u_idx = i * m3 + j
                                v_idx = j * p3 + k
                                w_idx = k * n3 + i
                                for r in range(R_target):
                                    val += U_list[u_idx][r] * V_list[v_idx][r] * W_list[w_idx][r]
                                if val != 1:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        return U_np, V_np, W_sol
                        
            except:
                pass
                
        return None

    # Try to find a decomposition for the 3x3x3 part with ranks 23, 24, 25...
    # Start with 23 (best known)
    for R_333 in range(23, 28):
        if time.time() - start_time > time_limit - 10:
            break
            
        res = find_rank_333x33(R_333, max_attempts=2000)
        if res is not None:
            U_333, V_333, W_333 = res
            # Construct the full 3x3x4 solution
            R_total = R_333 + 9
            
            U_full = np.zeros((n * m, R_total))
            V_full = np.zeros((m * p, R_total))
            W_full = np.zeros((p * n, R_total))
            
            # Copy 3x3x3 part (k=0,1,2)
            # U_full[i*3+j, 0:R_333] = U_333[i*3+j, 0:R_333]
            U_full[:, :R_333] = U_333
            # V_full[j*4+k, 0:R_333] for k=0,1,2
            for j in range(m):
                for k in range(3):
                    V_full[j * p + k, :R_333] = V_333[j * p3 + k, :]
            # W_full[k*3+i, 0:R_333] for k=0,1,2
            for k in range(3):
                for i in range(n):
                    W_full[k * n + i, :R_333] = W_333[k * p3 + i, :]
            
            # Add 3x3x1 part (k=3)
            # Ranks R_333 to R_333+8
            for i in range(n):
                for j in range(m):
                    r = R_333 + i * m + j
                    U_full[i * m + j, r] = 1
                    V_full[j * p + 3, r] = 1
                    W_full[3 * n + i, r] = 1
            
            # Verify full solution
            U_list = [[Fraction(x) for x in row] for row in U_full.tolist()]
            V_list = [[Fraction(x) for x in row] for row in V_full.tolist()]
            W_list = [[Fraction(x) for x in row] for row in W_full.tolist()]
            
            if is_valid(U_list, V_list, W_list):
                return {
                    "U": to_list(U_list),
                    "V": to_list(V_list),
                    "W": to_list(W_list)
                }

    # Fallback to Schoolbook
    rank = rank_schoolbook
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

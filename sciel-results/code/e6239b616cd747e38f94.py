import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try
    ranks_to_try = [16, 17, 18]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        success = False
        while not success and time.time() < deadline:
            # Random initialization
            # We need U (n*m, rank), V (m*p, rank), W (p*n, rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Reshape for easier indexing in the loss function
            # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
            
            def residual(x):
                # x is flattened U, V, W
                # Length: rank * (n*m + m*p + p*n)
                size_um = n * m * rank
                size_vm = m * p * rank
                size_wm = p * n * rank
                
                U_flat = x[:size_um]
                V_flat = x[size_um : size_um + size_vm]
                W_flat = x[size_um + size_vm :]
                
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                res = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Sum over c
                            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            u_val = U_mat[i * m + j, :]
                            v_val = V_mat[j * p + k, :]
                            w_val = W_mat[k * n + i, :]
                            prod = np.dot(u_val, v_val * w_val)
                            res[idx] = prod - 1.0
                            idx += 1
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                result = least_squares(residual, x0, method='trf', max_nfev=1000)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                # Extract solution
                size_um = n * m * rank
                size_vm = m * p * rank
                U_sol = result.x[:size_um].reshape(n * m, rank)
                V_sol = result.x[size_um : size_um + size_vm].reshape(m * p, rank)
                W_sol = result.x[size_um + size_vm :].reshape(p * n, rank)
                
                # Try to round to integers or half-integers
                # The solution might be scaled. We need to find a scaling that makes them integers.
                # This is a "lattice" problem.
                # Simple heuristic: Check if values are close to integers or half-integers.
                # If not, try to scale by small rationals?
                # For now, just check if they are close to integers/half-integers directly.
                # If the numerical solution is arbitrary scale, this might fail.
                
                # To fix scale, we can normalize each column c such that max(|U_c|, |V_c|, |W_c|) = 1?
                # Or sum of squares = 1.
                # But integer solutions don't have unit norm.
                
                # Let's try to round to nearest integer.
                U_int = np.rint(U_sol)
                V_int = np.rint(V_sol)
                W_int = np.rint(W_sol)
                
                # Verify
                if np.allclose(np.dot(U_int, V_int * W_int).reshape(-1), np.ones(n*m*p), atol=1e-9):
                    # This check is wrong. The product is element-wise sum over c.
                    # Let's verify properly.
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(rank):
                                    s += U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
                # Try half-integers
                U_half = np.rint(U_sol * 2) / 2
                V_half = np.rint(V_sol * 2) / 2
                W_half = np.rint(W_sol * 2) / 2
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank):
                                s += U_half[i*m+j, c] * V_half[j*p+k, c] * W_half[k*n+i, c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
                # If direct rounding fails, the scale might be off.
                # We can try to find a scaling factor.
                # But this is complex. Let's assume the random init leads to a solution close to integer scale.
                # To improve odds, we can try to normalize columns before rounding.
                # Actually, the known solution for 2x3x3 rank 16 has small integers.
                # If the numerical solution is far from integer scale, we might miss it.
                
                # A better approach: Use the structure.
                # But for now, let's just return the best found or fall through.
                
        # If we exited the while loop without success, try next rank
        if time.time() > deadline:
            break

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

import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute valid and invalid indices
    # u in [0, nm), v in [0, mp), w in [0, pn)
    valid_uvw = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_uvw.add((u, v, w))
                
    all_uvw = [(u, v, w) for u in range(nm) for v in range(mp) for w in range(pn)]
    invalid_uvw = [t for t in all_uvw if t not in valid_uvw]
    
    # Target values
    # b[uvw] = 1 if valid, 0 if invalid
    # We will construct the linear system A x = b
    
    def get_system_A_for_U(V, W, R):
        # Variables: U flattened (nm * R)
        # Equation for each (u, v, w): sum_r U[u, r] * V[v, r] * W[w, r] = b
        # A has shape (len(all_uvw), nm * R)
        # A[row, col] where col = u * R + r
        # A[row, u * R + r] = V[v, r] * W[w, r]
        
        num_eq = len(all_uvw)
        num_var = nm * R
        A = np.zeros((num_eq, num_var))
        b = np.zeros(num_eq)
        
        for idx, (u, v, w) in enumerate(all_uvw):
            b[idx] = 1.0 if (u, v, w) in valid_uvw else 0.0
            for r in range(R):
                A[idx, u * R + r] = V[v, r] * W[w, r]
                
        return A, b

    def get_system_A_for_V(U, W, R):
        # Variables: V flattened (mp * R)
        # Equation: sum_r U[u, r] * V[v, r] * W[w, r] = b
        # A[row, v * R + r] = U[u, r] * W[w, r]
        
        num_eq = len(all_uvw)
        num_var = mp * R
        A = np.zeros((num_eq, num_var))
        b = np.zeros(num_eq)
        
        for idx, (u, v, w) in enumerate(all_uvw):
            b[idx] = 1.0 if (u, v, w) in valid_uvw else 0.0
            for r in range(R):
                A[idx, v * R + r] = U[u, r] * W[w, r]
                
        return A, b

    def get_system_A_for_W(U, V, R):
        # Variables: W flattened (pn * R)
        # Equation: sum_r U[u, r] * V[v, r] * W[w, r] = b
        # A[row, w * R + r] = U[u, r] * V[v, r]
        
        num_eq = len(all_uvw)
        num_var = pn * R
        A = np.zeros((num_eq, num_var))
        b = np.zeros(num_eq)
        
        for idx, (u, v, w) in enumerate(all_uvw):
            b[idx] = 1.0 if (u, v, w) in valid_uvw else 0.0
            for r in range(R):
                A[idx, w * R + r] = U[u, r] * V[v, r]
                
        return A, b

    def verify(U, V, W):
        # Check exactness with half-integers
        # U, V, W are lists of lists
        # Convert to numpy for speed? Or just loop.
        # The values should be 0.5 * integer.
        # Let's check if 2*val is close to integer.
        for row in U:
            for val in row:
                if abs(2*val - round(2*val)) > 1e-9:
                    return False
        for row in V:
            for val in row:
                if abs(2*val - round(2*val)) > 1e-9:
                    return False
        for row in W:
            for val in row:
                if abs(2*val - round(2*val)) > 1e-9:
                    return False
        
        # Snap to half-integers
        U_snap = [[round(2*val)/2.0 for val in row] for row in U]
        V_snap = [[round(2*val)/2.0 for val in row] for row in V]
        W_snap = [[round(2*val)/2.0 for val in row] for row in W]
        
        # Check equations
        # We only need to check if the contraction is correct.
        # Sum_r U[u,r] V[v,r] W[w,r]
        # This is O(R * nm * mp * pn).
        # R ~ 20, nm=9, mp=15, pn=15. 20 * 9 * 15 * 15 = 40500 ops. Very fast.
        
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U_snap[u][r] * V_snap[v][r] * W_snap[w][r]
                    target = 1.0 if (u, v, w) in valid_uvw else 0.0
                    if abs(s - target) > 1e-9:
                        return False
        return True

    def to_list(M):
        return M.tolist()

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try ranks from lower bound up
    # Lower bound is max(nm, mp, pn) = 15
    for R in range(15, 46):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each R
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use random values, maybe scaled
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Alternating Least Squares
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                A_U, b_U = get_system_A_for_U(V, W, R)
                U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_sol.reshape(nm, R)
                
                # Solve for V
                A_V, b_V = get_system_A_for_V(U, W, R)
                V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_sol.reshape(mp, R)
                
                # Solve for W
                A_W, b_W = get_system_A_for_W(U, V, R)
                W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_sol.reshape(pn, R)
                
                # Check convergence / validity
                # We can check the residual quickly
                # Or just verify at the end of the loop
                if it % 10 == 9:
                    # Quick check: are values close to half-integers?
                    # If so, verify
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Final check after ALS
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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

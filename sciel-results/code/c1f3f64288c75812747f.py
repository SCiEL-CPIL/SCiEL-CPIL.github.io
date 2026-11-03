import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 145.0

    def _verify(U, V, W):
        rank = U.shape[1]
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Condition: sum_c W[k*n+i, c] U[i*m+j, c] V[jp*p+k, c] == delta(j, jp)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                        target = 1.0 if j == jp else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        return U_snapped, V_snapped, W_snapped

    def _als_solve(rank, max_iters=100):
        # Initialize
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # Build A and b for U[i*m+j]
                    # A is (p*m) x rank
                    # b is (p*m)
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for jp in range(m):
                            # Row k, jp
                            # Vector A_row: W[k*n+i, :] * V[jp*p+k, :] (elementwise)
                            a_row = W[k * n + i, :] * V[jp * p + k, :]
                            A_rows.append(a_row)
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    # Solve A u = b
                    u, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
            
            # Solve for V
            for jp in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for j in range(m):
                            a_row = W[k * n + i, :] * U[i * m + j, :]
                            A_rows.append(a_row)
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    v, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    V[jp * p + k, :] = v
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A_rows = []
                    b_vals = []
                    for j in range(m):
                        for jp in range(m):
                            a_row = U[i * m + j, :] * V[jp * p + k, :]
                            A_rows.append(a_row)
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    w, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
        
        # Check error
        error = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                        target = 1.0 if j == jp else 0.0
                        error += (s - target) ** 2
        
        if error < 1e-6:
            U_snapped, V_snapped, W_snapped = _snap_to_half_integers(U, V, W)
            if _verify(U_snapped, V_snapped, W_snapped):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
        return None

    # Try ranks 14, 15, 16
    for rank in [14, 15, 16]:
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            res = _als_solve(rank)
            if res is not None:
                return res
    
    return _schoolbook()

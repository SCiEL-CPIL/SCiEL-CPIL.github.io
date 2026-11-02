import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 110.0
    
    def compute_max_error(U, V, W):
        """Compute max absolute error over all (i,j,k) triples."""
        r = len(U[0])
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err
    
    def local_search(U, V, W, max_iter=5000):
        """Local search: try flipping entries to reduce max error."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        r = len(U[0])
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        cur_err = compute_max_error(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            if cur_err < 1e-9:
                break
            
            # Pick random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row = np.random.randint(n * m)
                col = np.random.randint(r)
                old_val = U[row][col]
                new_val = values[np.random.randint(len(values))]
                U[row][col] = new_val
                new_err = compute_max_error(U, V, W)
                if new_err > cur_err + 1e-12:
                    U[row][col] = old_val
                else:
                    cur_err = new_err
            elif mat_idx == 1:
                row = np.random.randint(m * p)
                col = np.random.randint(r)
                old_val = V[row][col]
                new_val = values[np.random.randint(len(values))]
                V[row][col] = new_val
                new_err = compute_max_error(U, V, W)
                if new_err > cur_err + 1e-12:
                    V[row][col] = old_val
                else:
                    cur_err = new_err
            else:
                row = np.random.randint(p * n)
                col = np.random.randint(r)
                old_val = W[row][col]
                new_val = values[np.random.randint(len(values))]
                W[row][col] = new_val
                new_err = compute_max_error(U, V, W)
                if new_err > cur_err + 1e-12:
                    W[row][col] = old_val
                else:
                    cur_err = new_err
        
        return U, V, W, cur_err
    
    # Try ranks from max(n,m,p) upwards
    for r in range(max(n, m, p), 25):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random starts with local search
        max_starts = 200
        for s in range(max_starts):
            if time.time() > deadline - 3.0:
                break
            
            # Random initialization
            U = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(r)] for _ in range(n * m)]
            V = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(r)] for _ in range(m * p)]
            W = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(r)] for _ in range(p * n)]
            
            # Local search
            U, V, W, cur_err = local_search(U, V, W, max_iter=2000)
            
            if cur_err < 1e-9:
                return {"U": U, "V": V, "W": W}
        
    # If not found, return schoolbook
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

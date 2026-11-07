import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = 290.0
    
    # Radau IIA coefficients (3 stages)
    c = np.array([0.2, 0.6, 1.0])
    A = np.array([
        [1/12, 0, 0],
        [1/12, 3/10, 0],
        [1/12, 3/10, 1/5]
    ])
    b = np.array([1/12, 3/10, 1/5])
    s = 3
    
    # Use N=100 steps for high accuracy
    N = 100
    h = (t1 - t0) / N
    t = t0
    
    for _ in range(N):
        if time.time() > deadline:
            break
            
        Y = np.tile(y, (s, 1))
        
        for newton_iter in range(15):
            F = np.zeros((s, n))
            f_stages = np.zeros((s, n))
            
            for j in range(s):
                t_j = t + c[j] * h
                f_stages[j] = f(t_j, Y[j])
            
            for i in range(s):
                F[i] = Y[i] - y - h * np.dot(A[i, :i+1], f_stages[:i+1])
                
            if np.max(np.abs(F)) < 1e-14:
                break
                
            J_blocks = np.zeros((s, s, n, n))
            J_f_stages = np.zeros((s, n, n))
            
            for j in range(s):
                J_f = np.zeros((n, n))
                for k in range(n):
                    e = np.zeros(n)
                    scale = 1e-8 * max(1.0, abs(Y[j, k]))
                    e[k] = scale
                    f_plus = f(t + c[j]*h, Y[j] + e)
                    J_f[:, k] = (f_plus - f_stages[j]) / scale
                J_f_stages[j] = J_f
                
            for i in range(s):
                for k in range(i + 1):
                    J_blocks[i, k] = (np.eye(n) if i == k else np.zeros((n, n))) - h * A[i, k] * J_f_stages[k]
            
            F_flat = F.flatten()
            J_flat = np.zeros((s*n, s*n))
            for i in range(s):
                for k in range(s):
                    J_flat[i*n:(i+1)*n, k*n:(k+1)*n] = J_blocks[i, k]
            
            try:
                delta = np.linalg.solve(J_flat, -F_flat)
            except np.linalg.LinAlgError:
                delta, _, _, _ = np.linalg.lstsq(J_flat, -F_flat, rcond=None)
                
            Y_flat = Y.flatten()
            Y_flat += delta
            Y = Y_flat.reshape(s, n)
            
        y = y + h * np.dot(b, f_stages)
        t += h
        
    return y

import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                T_recon[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def try_rank_reduction(U, V, W, target_rank, n=2, m=4, p=4, max_iterations=500):
    """Try to reduce rank by finding linear dependencies and removing redundant terms."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank
    
    norms = []
    for r in range(rank):
        norm = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
        norms.append(norm)
    
    order = sorted(range(rank), key=lambda r: norms[r])
    
    for r_to_remove in order:
        if time_up():
            break
        if current_rank <= target_rank:
            break
        
        mask = [i for i in range(current_rank) if i != r_to_remove]
        new_U = current_U[:, mask]
        new_V = current_V[:, mask]
        new_W = current_W[:, mask]
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank = len(mask)
            norms = []
            for r in range(current_rank):
                norm = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
                norms.append(norm)
            order = sorted(range(current_rank), key=lambda r: norms[r])
    
    if current_rank < rank:
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank
    
    return U, V, W, rank

def random_search(rank, n=2, m=4, p=4, iterations=1000):
    """Random search for a decomposition of given rank."""
    for it in range(iterations):
        if time_up():
            break
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist(), rank
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if not time_up():
        U2, V2, W2, rank2 = try_rank_reduction(U, V, W, 20, n, m, p)
        if rank2 < best_rank:
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    if best_result is None:
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")

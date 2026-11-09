import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def cost_vectorized(U, V, W):
        r = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    total += (val - 1.0) ** 2
        return total
    
    r = 14
    max_attempts = 80
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        strategy = attempt % 4
        
        if strategy == 0:
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for col in range(7):
                i = col % 2
                j = (col // 2) % 2
                k = (col // 4) % 2
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
            for col in range(7, 14):
                cc = col - 7
                i = cc % 2
                j = 2 + (cc // 2) % 3
                k = (cc // 6) % 3
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
        elif strategy == 1:
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for col in range(r):
                i = col % n
                j = (col // n) % m
                k = (col // (n*m)) % p
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
        elif strategy == 2:
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for col in range(r):
                for _ in range(2):
                    U[random.randint(0, n*m-1), col] = random.choice([-1, 1])
                    V[random.randint(0, m*p-1), col] = random.choice([-1, 1])
                    W[random.randint(0, p*n-1), col] = random.choice([-1, 1])
        else:
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        current_cost = cost_vectorized(U, V, W)
        
        temp = 1.0
        for _ in range(40):
            if time.time() > deadline - 5:
                break
            
            if current_cost < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            mat_idx = random.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = random.randint(0, mat.shape[0]-1)
            col = random.randint(0, mat.shape[1]-1)
            old_val = mat[row, col]
            mat[row, col] = old_val + random.choice([-2, 2])
            
            new_cost = cost_vectorized(U, V, W)
            
            if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / temp):
                current_cost = new_cost
            else:
                mat[row, col] = old_val
            
            temp *= 0.95
        
        if current_cost < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook
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

import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p, tol=1e-5):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    rng = np.random.RandomState(42)
    ones_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ones_set.add((i*m+j, j*p+k, k*n+i))
    for _ in range(300):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        c = rng.randint(0, p*n)
        if (a,b,c) not in ones_set:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val) > tol:
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, c] = 1.0
                V_sb[j*p+k, c] = 1.0
                W_sb[k*n+i, c] = 1.0
                c += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ones.append((i*m+j, j*p+k, k*n+i))
    ones_set = set(ones)
    
    for target_rank in [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 10:
            break
        
        for seed in range(15):
            if time.time() > deadline - 2:
                break
            
            np.random.seed(seed * 7919 + target_rank * 13)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            lr = 0.01
            for it in range(4000):
                if time.time() > deadline - 1:
                    break
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for (a, b, c) in ones:
                    val = np.sum(U[a] * V[b] * W[c])
                    diff = val - 1.0
                    gU[a] += 2*diff * V[b] * W[c]
                    gV[b] += 2*diff * U[a] * W[c]
                    gW[c] += 2*diff * U[a] * V[b]
                
                rng = np.random.RandomState(it*31 + seed)
                for _ in range(40):
                    a = rng.randint(0, n*m)
                    b = rng.randint(0, m*p)
                    c = rng.randint(0, p*n)
                    if (a,b,c) not in ones_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        gU[a] += 2*val * V[b] * W[c]
                        gV[b] += 2*val * U[a] * W[c]
                        gW[c] += 2*val * U[a] * V[b]
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                if it % 500 == 0 and it > 0:
                    lr *= 0.7
            
            if _verify(U, V, W, n, m, p, tol=1e-4):
                lr = 0.001
                for it in range(3000):
                    if time.time() > deadline - 1:
                        break
                    gU = np.zeros_like(U)
                    gV = np.zeros_like(V)
                    gW = np.zeros_like(W)
                    for (a, b, c) in ones:
                        val = np.sum(U[a] * V[b] * W[c])
                        diff = val - 1.0
                        gU[a] += 2*diff * V[b] * W[c]
                        gV[b] += 2*diff * U[a] * W[c]
                        gW[c] += 2*diff * U[a] * V[b]
                    U -= lr * gU
                    V -= lr * gV
                    W -= lr * gW
                    if it % 500 == 0:
                        lr *= 0.5
                
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank == target_rank:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

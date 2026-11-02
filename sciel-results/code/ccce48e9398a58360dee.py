import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine two block decompositions:
        - Block 1: M_{n,m,p1} with rank r1
        - Block 2: M_{n,m,p2} with rank r2
        Result: M_{n,m,p1+p2} with rank r1+r2
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank_total = r1 + r2
        p_total = p1 + p2
        
        U_full = [[0] * rank_total for _ in range(n * m)]
        V_full = [[0] * rank_total for _ in range(m * p_total)]
        W_full = [[0] * rank_total for _ in range(p_total * n)]
        
        # Copy block 1
        for i in range(n * m):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        
        for j in range(m):
            for k in range(p1):
                for r in range(r1):
                    V_full[j * p_total + k][r] = V1[j * p1 + k][r]
        
        for k in range(p1):
            for i in range(n):
                for r in range(r1):
                    W_full[k * n + i][r] = W1[k * n + i][r]
        
        # Copy block 2 (offset in rank dimension)
        for i in range(n * m):
            for r in range(r2):
                U_full[i][r1 + r] = U2[i][r]
        
        for j in range(m):
            for k in range(p2):
                for r in range(r2):
                    V_full[j * p_total + p1 + k][r1 + r] = V2[j * p2 + k][r]
        
        for k in range(p2):
            for i in range(n):
                for r in range(r2):
                    W_full[(p1 + k) * n + i][r1 + r] = W2[k * n + i][r]
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    def try_find_m332_random(rank_target, max_iter=10000, seed=42, n2=3, m2=3, p2=2):
        """
        Random search for integer/half-integer decomposition of M_{3,3,2}.
        """
        rng = np.random.RandomState(seed)
        
        # Precompute index arrays
        idx_U = np.array([[i * m2 + j for j in range(m2)] for i in range(n2)])
        idx_V = np.array([[j * p2 + k for k in range(p2)] for j in range(m2)])
        idx_W = np.array([[k * n2 + i for i in range(n2)] for k in range(p2)])
        
        for _ in range(max_iter):
            # Generate random sparse vectors with values in {-1, 0, 1}
            U = rng.randint(-1, 2, size=(n2 * m2, rank_target))
            V = rng.randint(-1, 2, size=(m2 * p2, rank_target))
            W = rng.randint(-1, 2, size=(p2 * n2, rank_target))
            
            # Verify
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = np.dot(U[idx_U[i,j], :], V[idx_V[j,k], :] * W[idx_W[k,i], :])
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        return None

    def try_find_m332_opt(rank_target, seed=42, n2=3, m2=3, p2=2, max_iter=2000):
        """
        Optimized search for M_{3,3,2} with snapping.
        """
        rng = np.random.RandomState(seed)
        
        idx_U = np.array([[i * m2 + j for j in range(m2)] for i in range(n2)])
        idx_V = np.array([[j * p2 + k for k in range(p2)] for j in range(m2)])
        idx_W = np.array([[k * n2 + i for i in range(n2)] for k in range(p2)])
        
        for attempt in range(5):
            U = rng.randn(n2 * m2, rank_target) * 0.5
            V = rng.randn(m2 * p2, rank_target) * 0.5
            W = rng.randn(p2 * n2, rank_target) * 0.5
            
            lr = 0.02
            for it in range(max_iter):
                errors = np.zeros((n2, m2, p2))
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            errors[i, j, k] = np.dot(U[idx_U[i,j], :], V[idx_V[j,k], :] * W[idx_W[k,i], :]) - 1.0
                
                loss = np.sum(errors ** 2)
                if loss < 1e-12:
                    break
                    
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            e = 2.0 * errors[i, j, k]
                            dU[idx_U[i,j], :] += e * V[idx_V[j,k], :] * W[idx_W[k,i], :]
                            dV[idx_V[j,k], :] += e * U[idx_U[i,j], :] * W[idx_W[k,i], :]
                            dW[idx_W[k,i], :] += e * U[idx_U[i,j], :] * V[idx_V[j,k], :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 100 == 0:
                    lr *= 0.95
            
            # Snap to integers
            U_snap = np.round(U)
            V_snap = np.round(V)
            W_snap = np.round(W)
            
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = np.dot(U_snap[idx_U[i,j], :], V_snap[idx_V[j,k], :] * W_snap[idx_W[k,i], :])
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        return None

    start_time = time.time()
    deadline = 140
    
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    sb333 = schoolbook(3, 3, 3)
    U333, V333, W333 = sb333["U"], sb333["V"], sb333["W"]
    
    # Try to find low rank M_{3,3,2}
    found_m332 = False
    for r2 in range(13, 18):
        if time.time() - start_time > deadline - 20:
            break
        
        res = try_find_m332_random(r2, max_iter=5000, seed=r2)
        if res:
            U2, V2, W2 = res
            result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
            if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                best_result = result
                best_rank = 27 + r2
                found_m332 = True
                break
        
        res = try_find_m332_opt(r2, seed=r2 + 100)
        if res:
            U2, V2, W2 = res
            result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
            if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                best_result = result
                best_rank = 27 + r2
                found_m332 = True
                break

    return best_result
# EVOLVE-BLOCK-END

import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Target tensor: T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p), dtype=float)
    
    def als_solve(U, V, W, rank, max_iters=500):
        """Run ALS and return (U, V, W, error)"""
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                return U, V, W, float('inf')
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            
            if time.time() > deadline - 5.0:
                return U, V, W, float('inf')
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M2 = np.zeros((n, rank))
                    for i in range(n):
                        M2[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M2.T, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            
            if time.time() > deadline - 5.0:
                return U, V, W, float('inf')
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M3 = np.zeros((m, rank))
                    for j in range(m):
                        M3[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M3.T, T[i, :, k], rcond=None)[0]
                    except:
                        pass
            
            if time.time() > deadline - 5.0:
                return U, V, W, float('inf')
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < 1e-12:
                return U, V, W, err
        
        return U, V, W, err
    
    def try_snap_and_verify(U, V, W, rank):
        """Try to snap to half-integers and verify"""
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_half[i * m + j, :], V_half[j * p + k, :] * W_half[k * n + i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < 1e-10:
            U_frac = [[Fraction(int(round(U_half[i, c] * 2))) / 2 for c in range(rank)] for i in range(n * m)]
            V_frac = [[Fraction(int(round(V_half[i, c] * 2))) / 2 for c in range(rank)] for i in range(m * p)]
            W_frac = [[Fraction(int(round(W_half[i, c] * 2))) / 2 for c in range(rank)] for i in range(p * n)]
            return {"U": U_frac, "V": V_frac, "W": W_frac}
        return None
    
    # Try ranks from 19 upward
    for rank in range(19, 25):
        if time.time() > deadline - 10.0:
            break
        
        # Random initializations
        for attempt in range(30):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(attempt * 1000 + rank * 100)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            U, V, W, err = als_solve(U, V, W, rank, max_iters=300)
            
            if time.time() > deadline - 10.0:
                break
            
            if err < 1e-10:
                result = try_snap_and_verify(U, V, W, rank)
                if result is not None:
                    return result
        
        # Structured initializations
        for attempt in range(20):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(attempt * 777 + rank * 13)
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                nU = np.random.choice(n * m, size=max(1, rank // 4), replace=False)
                nV = np.random.choice(m * p, size=max(1, rank // 4), replace=False)
                nW = np.random.choice(p * n, size=max(1, rank // 4), replace=False)
                U[nU, c] = np.random.choice([-1.0, 1.0], size=len(nU))
                V[nV, c] = np.random.choice([-1.0, 1.0], size=len(nV))
                W[nW, c] = np.random.choice([-1.0, 1.0], size=len(nW))
            
            U, V, W, err = als_solve(U, V, W, rank, max_iters=300)
            
            if time.time() > deadline - 10.0:
                break
            
            if err < 1e-10:
                result = try_snap_and_verify(U, V, W, rank)
                if result is not None:
                    return result
    
    # Fallback: schoolbook
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

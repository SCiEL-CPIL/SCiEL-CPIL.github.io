import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p, tol=1e-5):
    """
    Verify if U, V, W decompose the matrix multiplication tensor M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _als_decompose(n, m, p, rank, max_iters=500, seed=0, tol=1e-6, time_budget=None, start_time=None):
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V_r[j, k, :] * W_r[k, i, :]
                try:
                    U[i, j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U_r[i, j, :] * W_r[k, i, :]
                try:
                    V[j, k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_r[i, j, :] * V_r[j, k, :]
                try:
                    W[k, i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        if it % 10 == 0:
            if _verify(U, V, W, n, m, p, tol=1e-4):
                break

    if _verify(U, V, W, n, m, p, tol=tol):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Strategy 1: Tile a 2x3x3 decomposition
    # If M_{2,3,3} has rank r, then M_{2,3,9} can be formed by 3 disjoint blocks, rank 3r.
    # We search for M_{2,3,3} rank from 9 (lower bound m*p=9, n*m=6, p*n=6 -> max=9) up to 27.
    # Theoretical lower bound for M_{2,3,3} is 9.
    # If we find rank r for 2x3x3, the total rank is 3r.
    # For r=9, total 27. For r=10, total 30.
    
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a low rank decomposition for 2x3x3
    for r3 in range(9, 27):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            U3, V3, W3 = _als_decompose(2, 3, 3, r3, max_iters=100, seed=seed, tol=1e-6)
            if U3 is not None:
                # Verify
                if _verify(U3, V3, W3, 2, 3, 3, tol=1e-6):
                    # Tile to 2x3x9
                    total_rank = 3 * r3
                    if total_rank < best_rank:
                        U = np.zeros((n * m, total_rank))
                        V = np.zeros((m * p, total_rank))
                        W = np.zeros((p * n, total_rank))
                        
                        for k_block in range(3):
                            # Map 2x3x3 indices to 2x3x9 indices
                            # U[i*m + j, t] -> U[i*m + j, k_block * r3 + t]
                            # V[j*p + k, t] -> V[j*p + (k_block * 3 + k), k_block * r3 + t]
                            # W[k*n + i, t] -> W[(k_block * 3 + k)*n + i, k_block * r3 + t]
                            
                            for i in range(2):
                                for j in range(3):
                                    for t in range(r3):
                                        U[i * m + j, k_block * r3 + t] = U3[i * 3 + j, t]
                                        
                            for j in range(3):
                                for k in range(3):
                                    for t in range(r3):
                                        V[j * p + (k_block * 3 + k), k_block * r3 + t] = V3[j * 3 + k, t]
                                        
                            for k in range(3):
                                for i in range(2):
                                    for t in range(r3):
                                        W[(k_block * 3 + k) * n + i, k_block * r3 + t] = W3[k * 2 + i, t]
                        
                        if _verify(U, V, W, n, m, p, tol=1e-5):
                            best_rank = total_rank
                            best_U, best_V, best_W = U, V, W
                            found = True
                    break
        
        if found:
            # Found a valid tiling. Can we improve?
            # Lower bound for 2x3x9 is 27. If we found 27, we are optimal (likely).
            # If we found > 27, maybe try direct search for lower rank?
            # But tiling is robust.
            # Let's return this if it's good.
            # Actually, let's try to polish it or check if we can do better with direct search in remaining time?
            # Direct search for rank < 3r3 might be harder.
            # Given time, returning the tiling solution is a safe improvement over schoolbook (54).
            # Score = 54 / (3*r3).
            # If r3=9, score=2.0.
            # If r3=12, score=1.5.
            
            # Let's try to round to integers if possible?
            # For now, return the numerical solution.
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: Direct search for 2x3x9 if tiling failed (unlikely to be better than tiling)
    # But tiling from rank 9-26 for 2x3x3 should find something if ALS works.
    # If ALS failed completely, fall back to schoolbook.
    
    # One more attempt: Direct ALS for 2x3x9 with low rank
    # This is slow, so only if time remains.
    if time.time() < deadline - 10:
        for r in range(27, 40):
            U, V, W = _als_decompose(n, m, p, r, max_iters=50, seed=0, tol=1e-6)
            if U is not None and _verify(U, V, W, n, m, p, tol=1e-5):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


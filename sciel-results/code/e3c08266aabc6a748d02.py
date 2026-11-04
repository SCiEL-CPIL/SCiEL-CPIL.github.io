import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Build the 3x3x3 tensor T333
    T333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T333[a, b, c] = 1.0
    
    def _verify_full(U, V, W, n=3, m=3, p=6, tol=1e-5):
        """Verify full 3x3x6 decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape != (n*m, U.shape[1]): return False
        if V.shape != (m*p, V.shape[1]): return False
        if W.shape != (p*n, W.shape[1]): return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]: return False
        
        nm, mp, pn = n*m, m*p, p*n
        rank = U.shape[1]
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                    
        # Check zeros
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    def als_333_fast(rank, max_iter=300, tol=1e-10):
        """Fast ALS for 3x3x3 tensor decomposition using matrix operations."""
        # Precompute tensor as 3D array
        T = T333  # shape (9,9,9)
        
        U = np.random.randn(9, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(9, rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update W
            for r in range(rank):
                # Compute residual: R[a,b,c] = T[a,b,c] - sum_{s!=r} U[a,s]V[b,s]W[c,s]
                # R = T - sum_s U[:,s] V[:,s] W[:,s] + U[:,r] V[:,r] W[:,r]
                approx = np.einsum('as,bs,cs->abc', U, V, W)
                R = T - approx + np.einsum('as,bs,cs->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
                
                # W[:,r] = sum_{a,b} (U[a,r]*V[b,r]) * R[a,b,:] / sum_{a,b} (U[a,r]*V[b,r])^2
                coeff = U[:, r] * V[:, r]  # shape (9,9)
                numerator = np.einsum('ab,abc->c', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    W[:, r] = numerator / denominator
            
            # Update U
            for r in range(rank):
                approx = np.einsum('as,bs,cs->abc', U, V, W)
                R = T - approx + np.einsum('as,bs,cs->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
                
                coeff = V[:, r] * W[:, r]  # shape (9,9)
                numerator = np.einsum('bc,abc->a', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    U[:, r] = numerator / denominator
            
            # Update V
            for r in range(rank):
                approx = np.einsum('as,bs,cs->abc', U, V, W)
                R = T - approx + np.einsum('as,bs,cs->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
                
                coeff = U[:, r] * W[:, r]  # shape (9,9)
                numerator = np.einsum('ac,abc->b', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    V[:, r] = numerator / denominator
            
            # Compute error
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T - approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Verify
        if _verify_333_fast(U, V, W):
            return U, V, W
        return None
    
    def _verify_333_fast(U, V, W, tol=1e-5):
        """Verify 3x3x3 decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        for a in range(9):
            i_a = a // 3
            j_a = a % 3
            for b in range(9):
                j_b = b // 3
                k_b = b % 3
                for c in range(9):
                    k_c = c // 3
                    i_c = c % 3
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    # Try to find 3x3x3 decomposition starting from rank 23
    U333 = V333 = W333 = None
    for rank in range(23, 30):
        if time.time() > deadline - 15:
            break
        
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + rank * 104729)
            result = als_333_fast(rank, max_iter=200)
            if result is not None:
                U333, V333, W333 = result
                break
        
        if U333 is not None:
            break
    
    # Build 3x3x6 decomposition from 3x3x3
    if U333 is not None:
        rank_333 = U333.shape[1]
        rank_total = 2 * rank_333  # Two 3x3x3 blocks for k=0,1,2 and k=3,4,5
        
        U = np.zeros((9, rank_total))
        V = np.zeros((18, rank_total))
        W = np.zeros((18, rank_total))
        
        # First half: k in [0,3)
        for r in range(rank_333):
            U[:, r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + k, r] = V333[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W[k*3 + i, r] = W333[k*3 + i, r]
        
        # Second half: k in [3,6)
        for r in range(rank_333):
            U[:, rank_333 + r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + (k+3), rank_333 + r] = V333[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3 + i, rank_333 + r] = W333[k*3 + i, r]
        
        # Verify
        if _verify_full(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: schoolbook rank 54
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

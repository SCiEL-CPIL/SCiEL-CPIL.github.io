"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # The target tensor T[i, j, k] = 1 for all (i,j,k) in the multiplication pattern
    # Actually, T is defined such that C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor entries: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the target tensor as a 3D array indexed by (a, b, c) where:
    # a = i*m + j (index into U rows)
    # b = j*p + k (index into V rows)  
    # c = k*n + i (index into W rows)
    # T[a, b, c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p  # 32
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions using randomized optimization
    # We'll try ranks from 20 up to 31
    for rank in range(20, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Optimize using gradient descent
            lr = 0.01
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor estimate
                # T_est[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # We only need to check the non-zero entries of target
                # Get all non-zero positions
                nonzero = np.argwhere(target != 0)
                
                # For efficiency, compute residual only at non-zero positions
                # But for gradient, we need full tensor or use sampling
                # Let's use a simpler approach: compute full tensor and compute gradient
                
                # T_est = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)
                # This is expensive for large dimensions, but our dims are small
                T_est = np.zeros((nm, mp, pn), dtype=np.float64)
                for r in range(rank):
                    T_est += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                residual = T_est - target
                loss = np.sum(residual ** 2)
                
                if loss < 1e-6:
                    # Found a good solution
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
                
                # Compute gradients
                # dL/dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[b,r] = 2 * sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[c,r] = 2 * sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(rank):
                    # grad_U[:,r] = 2 * einsum('abc,bc->a', residual, V[:,r]*W[:,r])
                    # Actually: sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                    VW = V[:,r] * W[:,r]  # shape (mp, pn) -> no, V is (mp,rank), W is (pn,rank)
                    # V[:,r] has shape (mp,), W[:,r] has shape (pn,)
                    # So V[:,r] * W[:,r] doesn't work directly. We need outer product
                    # sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                    # = einsum('abc,b,c->a', residual, V[:,r], W[:,r])
                    grad_U[:,r] = 2.0 * np.einsum('abc,b,c->a', residual, V[:,r], W[:,r])
                    grad_V[:,r] = 2.0 * np.einsum('abc,a,c->b', residual, U[:,r], W[:,r])
                    grad_W[:,r] = 2.0 * np.einsum('abc,a,b->c', residual, U[:,r], V[:,r])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check final loss
            T_est = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                T_est += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            residual = T_est - target
            loss = np.sum(residual ** 2)
            
            if loss < 1e-6:
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

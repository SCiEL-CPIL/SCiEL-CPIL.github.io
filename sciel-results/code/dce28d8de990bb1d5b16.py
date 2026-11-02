"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Verify decomposition
    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(S - T))
        return err < tol
    
    # Strassen-based decomposition for (3,4,8)
    # Strategy: Split m=4 into two 2-dimensions, use Strassen on 2x2x2 cores
    # 
    # A is 3x4, B is 4x8, C is 3x8
    # Split: A = [A_left A_right] (3x2 each), B = [B_top; B_bottom] (2x8 each)
    # C = A_left * B_top + A_right * B_bottom
    #
    # For each (3,2,8) sub-problem, we can further decompose.
    # (3,2,8): A is 3x2, B is 2x8, C is 3x8
    # C_i,k = A_i,0 * B_0,k + A_i,1 * B_1,k
    # This is schoolbook with rank 3*2*8 = 24 per sub-problem, total 48.
    # But we can do better using Strassen-like techniques on the 2x2x2 core.
    
    # Let's build the decomposition using the tensor product structure.
    # The (n,m,p) tensor T satisfies:
    # T[i,j,k] = 1 if indices match the multiplication pattern
    # 
    # For m=4, split into m1=2, m2=2:
    # T_{3,4,8}[i, j, k] where j in 0..3
    # If j in 0..1 (first half): T[i, j, k] = T_{3,2,8}[i, j, k]
    # If j in 2..3 (second half): T[i, j-2, k] = T_{3,2,8}[i, j-2, k]
    #
    # So the (3,4,8) tensor is the "sum" (in tensor rank sense) of two (3,2,8) tensors.
    # rank(T_{3,4,8}) <= rank(T_{3,2,8}) + rank(T_{3,2,8}) = 2 * rank(T_{3,2,8})
    
    # Now for (3,2,8): A is 3x2, B is 2x8, C is 3x8
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Schoolbook: rank = 3*2*8 = 24
    # 
    # Can we reduce this? The (3,2,8) tensor has structure:
    # T[i, j, k] = 1 for all i in 0..2, j in 0..1, k in 0..7
    # This is actually a simple outer product structure!
    # T = sum_{j=0}^{1} (e_i) (e_j) (e_k) for all i,j,k
    # Wait, that's just the all-ones tensor of shape (3,2,8), rank 1?
    # No! The matrix multiplication tensor is NOT all ones.
    
    # Let me re-examine. T[i*m+j, j*p+k, k*n+i] = 1
    # For (3,2,8): n=3, m=2, p=8
    # T[i*2+j, j*8+k, k*3+i] = 1
    # i in 0..2, j in 0..1, k in 0..7
    # First index: i*2+j in 0..5
    # Second index: j*8+k in 0..15
    # Third index: k*3+i in 0..23
    #
    # This is the standard matrix multiplication tensor. It's not all ones.
    
    # Let me just build the schoolbook decomposition and try to reduce it.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try Strassen-inspired decomposition
    # For (3,4,8), split m=4 into m1=2, m2=2
    # C = A1*B1 + A2*B2
    # where A1 is 3x2, B1 is 2x8, A2 is 3x2, B2 is 2x8
    #
    # For each (3,2,8) sub-problem, use schoolbook: rank 24
    # Total: 48
    
    def strassen_split():
        """Split (3,4,8) into two (3,2,8) sub-problems"""
        sub_rank = n * 2 * p  # 3*2*8 = 24 per sub-problem
        total_rank = 2 * sub_rank  # 48
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Sub-problem 1: j in 0..1 (first half of m)
        col = 0
        for i in range(n):
            for j in range(2):  # j=0,1
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Sub-problem 2: j in 2..3 (second half of m)
        for i in range(n):
            for j in range(2, 4):  # j=2,3
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W
    
    U_s, V_s, W_s = strassen_split()
    if verify(U_s, V_s, W_s):
        if U_s.shape[1] < best_rank:
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    # Now try to reduce each (3,2,8) sub-problem further
    # For (3,2,8), can we use a lower-rank decomposition?
    # The (3,2,8) tensor: T[i*2+j, j*8+k, k*3+i] = 1
    # This has rank at most 3*2*8 = 24 by schoolbook.
    # 
    # Actually, for (n,2,p) where n=3, p=8:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a bilinear map from (R^3 x R^2) x (R^2 x R^8) -> R^(3x8)
    # 
    # The tensor has the structure of a "slice" of the full multiplication tensor.
    # Let me check if there's a known lower bound or algorithm.
    
    # For now, let's try random optimization to reduce the rank further.
    for target_rank in range(best_rank - 1, max(best_rank - 30, 40), -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Normalize
            for c in range(target_rank):
                nu = np.linalg.norm(U_r[:, c])
                nv = np.linalg.norm(V_r[:, c])
                nw = np.linalg.norm(W_r[:, c])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    scale = (nu * nv * nw) ** (-1/3)
                    U_r[:, c] *= scale
                    V_r[:, c] *= scale
                    W_r[:, c] *= scale
            
            lr = 0.1
            for step in range(300):
                if time.time() > deadline - 30:
                    break
                
                S = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                diff = S - T
                err = np.sum(diff**2)
                
                if err < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
                grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
                grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
                
                nU = np.linalg.norm(grad_U)
                nV = np.linalg.norm(grad_V)
                nW = np.linalg.norm(grad_W)
                if nU > 1e-10: grad_U /= nU
                if nV > 1e-10: grad_V /= nV
                if nW > 1e-10: grad_W /= nW
                
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                for c in range(target_rank):
                    nu = np.linalg.norm(U_r[:, c])
                    nv = np.linalg.norm(V_r[:, c])
                    nw = np.linalg.norm(W_r[:, c])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        scale = (nu * nv * nw) ** (-1/3)
                        U_r[:, c] *= scale
                        V_r[:, c] *= scale
                        W_r[:, c] *= scale
                
                lr *= 0.99
                if lr < 1e-4:
                    break
            
            if best_rank == target_rank:
                break
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

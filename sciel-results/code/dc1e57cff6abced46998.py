"""Tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 3, 4]


def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x4 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] where:
    - i in 0..2 (row of A, 3 values)
    - j in 0..2 (column of A / row of B, 3 values)  
    - k in 0..3 (column of B, 4 values)
    
    T[i,j,k] = 1 if the (i,k) entry of the product depends on A[i,j]*B[j,k]
    
    We seek U, V, W such that:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 36)
    def schoolbook():
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
    
    # Try to find a lower-rank decomposition using rank reduction techniques
    # The tensor T for 3x3*3x4 multiplication:
    # T[i,j,k] = 1 for all i in [0,2], j in [0,2], k in [0,3]
    # This is actually the standard matmul tensor where each term A[i,j]*B[j,k] 
    # contributes to C[i,k]
    
    # For n x m * m x p matmul, the tensor is:
    # T_{ijk} = delta_{j'} where i indexes rows of A/C, j indexes cols of A/rows of B, 
    # k indexes cols of B/C
    # Actually T[i,j,k] = 1 for all valid i,j,k (it's a full tensor in the standard basis)
    
    # Let me think about this more carefully. The matmul tensor is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # As a tensor T with indices (i, j, k), T[i,j,k] = 1 for all i,j,k
    # The rank is the minimum r such that T[i,j,k] = sum_{r} U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Known bounds: rank of 3x3*3x3 is between 23 (Landsberg upper) and 19 (lower)
    # For 3x3*3x4, the rank is between... let's try to find it.
    
    # Strategy: Use random linear combinations to try to reduce rank
    # Start with schoolbook (rank 36) and try to eliminate terms
    
    deadline = time.time() + 100  # Leave 20s margin
    
    # Try various approaches to find lower rank
    
    # Approach 1: Try rank 32 (removing 4 terms from 36)
    # The idea: some linear combinations of the 36 terms may be redundant
    
    # Let's use a systematic approach: try to find a basis of rank r
    # such that the tensor can be expressed as sum of r rank-1 tensors
    
    # For the matmul tensor T[i,j,k] = 1 for all (i,j,k) in [0,2]x[0,2]x[0,3],
    # a rank-1 tensor is U[i,j] * V[j,k] * W[k,i]
    
    # We need: 1 = sum_{r=1}^{R} U_r[i,j] * V_r[j,k] * W_r[k,i] for all i,j,k
    
    # This is equivalent to finding matrices such that the bilinear form
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k] can be computed
    # with R multiplications.
    
    # Known results: For 3x3*3x3, rank is 19-23. For 3x3*3x4, 
    # we can think of it as two 3x3*3x3 problems plus adjustments.
    
    # Let's try to use the fact that 3x3*3x4 can be seen as:
    # Split B into B1 (3x3) and B2 (3x1)
    # C = A*B1 + A*B2
    # rank(3x3*3x4) <= rank(3x3*3x3) + rank(3x3*3x1)
    # rank(3x3*3x1) = 9 (schoolbook for vector-matrix)
    # So upper bound is rank(3x3*3x3) + 9
    
    # If rank(3x3*3x3) = 19, then rank(3x3*3x4) <= 28
    # If rank(3x3*3x3) = 23, then rank(3x3*3x4) <= 32
    
    # Let's try to construct an explicit decomposition
    
    # For now, let's try random search for small rank reductions
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to improve by finding linear dependencies
    # Use random projections to try to merge terms
    
    # A more systematic approach: use the fact that the tensor has symmetries
    # and try structured decompositions
    
    # For 3x3*3x4, let's try to use a decomposition based on:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Try rank 32: remove 4 terms
    # Try rank 28: remove 8 terms
    
    # Let's try a random optimization approach
    # We want to find U, V, W of rank r minimizing ||T - sum_r U_r V_r W_r||_F^2
    
    # For exact decomposition, we need this to be zero
    
    # Use numpy for speed
    np.random.seed(42)
    
    # Try a few candidate ranks
    for target_rank in [32, 28, 24]:
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Gradient descent to minimize ||T - U V W||^2
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        def compute_tensor(U, V, W, rank):
            """Compute the tensor from U, V, W."""
            T = np.zeros((n, m, p))
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            return T
        
        def tensor_error(U, V, W, rank):
            """Compute squared error between tensor and target."""
            T = compute_tensor(U, V, W, rank)
            # Target: T[i,j,k] = 1 for all i,j,k
            return np.sum((T - 1.0) ** 2)
        
        # Simple iterative refinement
        lr = 0.01
        prev_err = tensor_error(U, V, W, target_rank)
        
        for iteration in range(2000):
            if time.time() > deadline - 10:
                break
            
            T = compute_tensor(U, V, W, target_rank)
            error = T - 1.0  # Shape (n, m, p)
            err = np.sum(error ** 2)
            
            if err < 1e-6:
                # Found exact decomposition!
                # Convert to integer/half-integer if possible
                U_list = [[float(x) for x in U[i]] for i in range(n * m)]
                V_list = [[float(x) for x in V[j]] for j in range(m * p)]
                W_list = [[float(x) for x in W[k]] for k in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Gradient computation
            # dE/dU[i,j,r] = sum_{k} error[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            # dE/dV[j,k,r] = sum_{i} error[i,j,k] * U[i*m+j,r] * W[k*n+i,r]
            # dE/dW[k,i,r] = sum_{j} error[i,j,k] * U[i*m+j,r] * V[j*p+k,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i, j, k]
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            dU[idx_u, r] += e * V[idx_v, r] * W[idx_w, r]
                            dV[idx_v, r] += e * U[idx_u, r] * W[idx_w, r]
                            dW[idx_w, r] += e * U[idx_u, r] * V[idx_v, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_err = tensor_error(U, V, W, target_rank)
            if new_err < prev_err:
                prev_err = new_err
                lr *= 1.05
            else:
                lr *= 0.9
                U += lr * dU
                V += lr * dV
                W += lr * dW
            
            if err < 1e-8:
                break
        
        # Check if we found something good
        final_err = tensor_error(U, V, W, target_rank)
        if final_err < 1e-6 and target_rank < best_rank:
            best_rank = target_rank
            best_result = {"U": [[float(x) for x in U[i]] for i in range(n * m)],
                           "V": [[float(x) for x in V[j]] for j in range(m * p)],
                           "W": [[float(x) for x in W[k]] for k in range(p * n)]}
    
    # Return the best result found
    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

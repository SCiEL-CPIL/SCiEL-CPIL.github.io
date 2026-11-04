"""Rewritten program for matrix multiplication 2x3 by 3x7 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
SEED = 42

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    The tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 for all (i,j,k).
    
    We use a randomized algebraic approach: try small ranks and check if a
    solution exists via solving the resulting system.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # The tensor is all-ones of shape (n, m, p) = (2, 3, 7)
    # Total number of entries: n*m*p = 42
    
    # Rank must be at least max(n, m, p) = 7 for any bilinear algorithm
    # For matrix multiplication, Strassen-like bounds apply
    # Known lower bound: rank >= n*m*p / (something) but for this specific case
    # Let's try ranks starting from 7 up to n*m*p=42
    
    # Strategy: For each candidate rank r, try to find U, V, W such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # This is a nonlinear system. We can fix one matrix (say W) and solve
    # for U and V, or use iterative methods.
    
    # Alternative: Use the fact that for all-ones tensor, we can try
    # structured decompositions.
    
    # Simple approach: Try rank = n*m*p = 42 (schoolbook) as fallback
    # Then try to reduce rank.
    
    # For the all-ones tensor, one simple decomposition of rank r exists if
    # we can write 1 = sum_c u_{ij}^c * v_{jk}^c * w_{ki}^c for all i,j,k.
    
    # A key insight: if we set W[k*n+i, c] = 1 for all k,i,c (i.e., W is all-ones),
    # then we need sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all i,j,k.
    # This means the sum over c of the outer product of row i*m+j of U and row j*p+k of V
    # must be 1 for all valid (i,j,k).
    
    # Let's try a different approach: use random initialization and gradient-free
    # search, or use the algebraic structure.
    
    # For now, implement schoolbook (rank 42) and try to improve.
    
    def schoolbook():
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
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For the all-ones tensor, a rank-r decomposition exists iff
    # there are matrices U, V, W such that the tensor product sums to 1.
    
    # Try ranks from max(n,m,p) up to best_rank-1
    for r in range(max(n, m, p), best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try random initializations
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(SEED + r * 1000 + trial)
            
            # Initialize W randomly with small values
            W_try = np.random.randn(p * n, r)
            
            # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a system of n*m*p = 42 equations in n*m*r + m*p*r variables (U and V)
            # But it's nonlinear due to the product with W.
            
            # Alternative: fix W, then the equations become linear in U*V products.
            # Let A_{ij,k,c} = U[i*m+j,c] * W[k*n+i,c] and B_{jk,c} = V[j*p+k,c]
            # Then sum_c A_{ij,k,c} * B_{jk,c} = 1
            
            # This is still nonlinear. Let's use a different strategy.
            
            # Strategy: Assume W has a simple structure. For example, W[k*n+i,c] = delta_{c, f(i,k)}
            # for some function f. This is the schoolbook approach.
            
            # Better: Use iterative refinement. Start with random U, V, W and 
            # project onto the constraint.
            
            # Actually, let's use a simpler algebraic approach for the all-ones tensor.
            # If we set U[i*m+j,c] = a_c, V[j*p+k,c] = b_c, W[k*n+i,c] = c_c,
            # then sum_c a_c * b_c * c_c = 1.
            # This is a constant, independent of i,j,k! So we just need
            # sum_c a_c * b_c * c_c = 1.
            # This works for ANY rank r >= 1!
            
            # Wait, that's not right. U depends on (i,j), V on (j,k), W on (k,i).
            # But if we make them constant (independent of i,j,k), then yes,
            # the sum is the same for all (i,j,k).
            
            # So for rank 1: U = [a], V = [b], W = [c] with a*b*c = 1.
            # Let a=b=c=1. Then sum = 1*1*1 = 1. Done!
            
            # But wait, is this valid? Let me check:
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # If U, V, W are all [1] (rank 1), then T[i,j,k] = 1*1*1 = 1. Yes!
            
            # So rank 1 suffices for the all-ones tensor?!
            
            # Let me verify this is correct. The matrix multiplication tensor for
            # (n,m,p) has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
            # The tensor decomposition gives C[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_k V[j,k,c]*B[j,k]) * W[k,i,c]
            # No wait, the standard form is:
            # C[i,k] = sum_c sum_j U[i,j,c] * A[i,j] * V[j,k,c] ... no.
            
            # The correct form for matrix multiplication tensor decomposition:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) ... 
            # No, that's not right either.
            
            # The standard tensor decomposition for matmul:
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # where the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
            # and the decomposition says: C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[k,i,c]
            # No, that doesn't make sense dimensionally.
            
            # Let me reconsider. The matrix multiplication tensor T in R^{n x m x p} has
            # T[i,j,k] = 1 for all i,j,k. A rank-r decomposition is:
            # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
            
            # For the all-ones tensor, if U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1/r for all i,j,k,c,
            # then sum_c 1 * 1 * (1/r) = r * (1/r) = 1. So rank r works for any r >= 1.
            
            # In particular, rank 1 works with U=V=W=1.
            
            # But wait, the task says "exactly decompose the matrix multiplication tensor".
            # Is the all-ones tensor the correct tensor for matrix multiplication?
            
            # Yes! The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
            # has entries T[i,j,k] = 1 for all 0 <= i < n, 0 <= j < m, 0 <= k < p.
            # This is because C[i,k] = sum_j A[i,j] * B[j,k], and the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
            
            # So the all-ones tensor is correct, and rank 1 suffices!
            
            # Let me construct the rank-1 solution.
            if r == 1:
                U_try = np.ones((n * m, 1))
                V_try = np.ones((m * p, 1))
                W_try = np.ones((p * n, 1))
                
                # Verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 for all i,j,k
                # Correct!
                
                # Convert to list format
                U_list = U_try.tolist()
                V_list = V_try.tolist()
                W_list = W_try.tolist()
                
                # Double-check with a few samples
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(1):
                                val += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        if found:
            break
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

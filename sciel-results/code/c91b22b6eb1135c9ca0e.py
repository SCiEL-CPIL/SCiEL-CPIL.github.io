"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# Known rank bounds for 2x4 * 4x7 matrix multiplication
# Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
# Upper bound: n*m*p = 56 (schoolbook)
# Known results suggest rank is likely around 30-40 for these dimensions

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    schoolbook_rank = n * m * p  # 56
    
    # Start with schoolbook as fallback
    def schoolbook_decomposition():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition()
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_sb
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k,l] = 1 if the (i,j) element of A times (j,k) element of B 
        # contributes to (i,k) element of C
        # Actually, for matrix multiplication C = A*B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle dimension)
        # 
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-read the convention: W[k*n+i, r]
        # So W is indexed as W[k*n+i, r] where i is row of A (0..n-1), k is col of B (0..p-1)
        # U is indexed as U[i*m+j, r] where i is row of A, j is col of A / row of B
        # V is indexed as V[j*p+k, r] where j is row of B, k is col of B
        #
        # The product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # for all valid i, j, k (since T[i,j,k] = 1 for matrix multiplication)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random initialization and optimization
    # Strategy: Try various ranks from lower bound upward
    
    # Lower bound is max(n*m, m*p, n*p) = max(8, 28, 14) = 28
    # But we can also use the Strassen-like lower bound
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Use random projections and check if we can find a valid decomposition
    
    def try_rank(rank, max_iterations=5000, seed=None):
        """Try to find a decomposition of given rank using random search."""
        if rank < lower_bound:
            return None
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to make the product closer to 1
        # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # This is a system of n*m*p equations with 3*rank variables
        # n*m*p = 56, 3*rank variables
        
        # Use a simple iterative approach: adjust one factor at a time
        # This is essentially trying to solve a nonlinear system
        
        # Alternative: Use the fact that we can fix some structure
        # For small dimensions, we can try to find integer/rational solutions
        
        # Let's try a different approach: start with schoolbook and try to merge terms
        # by finding linear dependencies
        
        return None
    
    # Better approach: Use known algebraic structures
    # For 2x4 * 4x7, we can try to exploit the small dimension n=2
    
    # The tensor is T[i,j,k] = delta_{i,k} * 1 (for the multiplication structure)
    # Actually no: T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor component is 1
    
    # Let me try a structured approach:
    # Since n=2 is small, we can parameterize more carefully
    
    # Try rank 28 (the theoretical lower bound)
    # This requires 28 rank-1 terms to represent 56 tensor components
    
    # Use random initialization with constraints
    def find_decomposition_for_rank(target_rank, num_trials=100, seed=None):
        """Try to find an exact decomposition for a given target rank."""
        if seed is not None:
            np.random.seed(seed)
        
        n, m, p = DIMENSIONS
        num_tensor_components = n * m * p  # 56
        
        for trial in range(num_trials):
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to get closer to the target
            # For each (i,j,k), compute the current product
            # and try to adjust
            
            # Use gradient descent on the squared error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            
            lr = 0.01
            for it in range(2000):
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # product = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                            prod = 0
                            for r in range(target_rank):
                                prod += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = prod - 1
                            
                            for r in range(target_rank):
                                dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if it % 500 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                prod = 0
                                for r in range(target_rank):
                                    prod += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                max_err = max(max_err, abs(prod - 1))
                    if max_err < 1e-12:
                        # Found a solution! Now try to round to rationals
                        # For now, return the floating point solution
                        return U, V, W
            
            # Check if we got close
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = 0
                        for r in range(target_rank):
                            prod += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(prod - 1))
            
            if max_err < 1e-10:
                return U, V, W
        
        return None
    
    # Try various ranks from lower bound up
    # But this is computationally expensive, so be strategic
    
    # Start with ranks that are more likely to work
    # For 2x4*4x7, known results suggest rank around 30-40
    
    candidate_ranks = []
    for r in range(lower_bound, min(schoolbook_rank, 45)):
        candidate_ranks.append(r)
    
    random.shuffle(candidate_ranks)
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = find_decomposition_for_rank(target_rank, num_trials=20, seed=42)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, target_rank):
                # Found a valid decomposition!
                # Try to convert to rational numbers
                best_rank = target_rank
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                print(f"Found decomposition with rank {target_rank}")
                # Don't break; try to find even lower rank
                continue
    
    # If we found a better rank, use it; otherwise use schoolbook
    return best_result

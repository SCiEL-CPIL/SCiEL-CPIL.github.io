import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute target tensor
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def check_solution(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, TARGET_TENSOR, atol=1e-8)

    def try_random_search(rank, max_iters=50000, seed=42):
        """
        Random search over integer/half-integer entries.
        We search for U, V, W with entries in {0, ±1, ±2, ±3, ±4} / 2
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Search space: entries in {-2, -1, 0, 1, 2} (i.e., half-integers -1, -0.5, 0, 0.5, 1)
        # But also allow integers -2, -1, 0, 1, 2 for more flexibility
        # Let's use a sparse representation approach
        
        # For efficiency, we'll use a different strategy:
        # Start with random sparse matrices and refine
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random sparse initialization
            density = np.random.uniform(0.1, 0.5)
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Fill with random half-integers
            for _ in range(int(n*m*rank*density)):
                r_idx = np.random.randint(n*m)
                c_idx = np.random.randint(rank)
                U[r_idx, c_idx] = np.random.choice([-2, -1, 0, 1, 2]) / 2.0
            
            for _ in range(int(m*p*rank*density)):
                r_idx = np.random.randint(m*p)
                c_idx = np.random.randint(rank)
                V[r_idx, c_idx] = np.random.choice([-2, -1, 0, 1, 2]) / 2.0
            
            for _ in range(int(p*n*rank*density)):
                r_idx = np.random.randint(p*n)
                c_idx = np.random.randint(rank)
                W[r_idx, c_idx] = np.random.choice([-2, -1, 0, 1, 2]) / 2.0
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_approx - TARGET_TENSOR)**2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    def try_gradient_descent(rank, max_iters=1000, seed=42):
        """
        Gradient descent with projection onto half-integers.
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - TARGET_TENSOR
            
            # Gradients
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Project to half-integers
            U = np.rint(U * 2) / 2.0
            V = np.rint(V * 2) / 2.0
            W = np.rint(W * 2) / 2.0
            
            err = np.sum(diff**2)
            if err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Decay learning rate
            lr *= 0.99
        
        return None

    def try_algebraic_search(rank):
        """
        More structured search using the algebraic structure of matrix multiplication.
        """
        if time.time() > deadline:
            return None
        
        # For matrix multiplication, we know the tensor has a specific structure.
        # Let's try to find a decomposition by solving a system of equations.
        
        # The decomposition is:
        # (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all i, j, k
        
        # This gives n*m*p equations with 3*rank variables (but they're coupled).
        
        # Let's use a different approach: fix U and V, solve for W.
        # W[k*n+i, r] = sum_{j} (A*B)[i,k] / (U[i*m+j, r] * V[j*p+k, r])
        # This doesn't work directly because of the product.
        
        # Instead, let's use a randomized approach with structure.
        
        np.random.seed(12345 + rank)
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Create structured random matrices
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Use block structure
            for r in range(rank):
                # Each rank-1 component should contribute to some (i,j,k) triple
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U[i*m+j, r] = np.random.choice([-1, 1])
                V[j*p+k, r] = np.random.choice([-1, 1])
                W[k*n+i, r] = np.random.choice([-1, 1])
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_approx - TARGET_TENSOR)**2)
            
            if err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Search ranks from low to high
    # Known upper bound for 3x3x4 is 36 (schoolbook).
    # Literature suggests ranks around 24-28.
    for r in range(24, 36):
        if time.time() > deadline:
            break
        
        # Try multiple strategies
        for strategy in [try_random_search, try_gradient_descent, try_algebraic_search]:
            if time.time() > deadline:
                break
            result = strategy(r)
            if result is not None:
                return result
    
    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}

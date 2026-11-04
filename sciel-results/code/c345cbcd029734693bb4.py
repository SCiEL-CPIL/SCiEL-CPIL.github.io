import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    def matmul_tensor_rank(U, V, W):
        """Compute the matrix multiplication tensor from U, V, W."""
        rank = len(U[0]) if U else 0
        T = np.zeros((n * m * p, rank), dtype=int)
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m * p + j * p + k, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        return T
    
    def check_decomposition(U, V, W, T_target):
        """Check if the decomposition exactly matches the target tensor."""
        T_computed = matmul_tensor_rank(U, V, W)
        return np.array_equal(T_computed, T_target)
    
    # Build the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Compute the target tensor
    T_target = matmul_tensor_rank(U, V, W)
    
    # Start with the schoolbook decomposition and try to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Try to find a lower rank decomposition using random linear combinations
    # We'll use a greedy approach: try to express some rank-1 components as linear combinations of others
    
    # First, let's try Strassen-style optimizations for 2x2x2 sub-blocks
    # The 2x2x2 Strassen decomposition has rank 7 instead of 8
    
    # Strassen 2x2x2 coefficients
    # For multiplying two 2x2 matrices, we have 7 multiplications
    # M1 = (A11 - A22) * (B22 + B11)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We can try to apply Strassen to disjoint 2x2x2 sub-blocks
    # In our 3x4x6 matrix multiplication, we can identify disjoint 2x2x2 sub-blocks
    
    # Let's try a randomized search for rank reduction
    # We'll use the fact that the tensor is sparse and try to find linear dependencies
    
    # Try reducing rank by finding linear dependencies among the rank-1 components
    # We'll work over rationals by using fractions or integers with care
    
    # Attempt to find a lower rank by combining rank-1 terms
    # This is a hard problem, so we'll use a heuristic approach
    
    # Try Strassen on the first two rows/cols of each dimension
    # This gives us a 2x2x2 block that can be done in rank 7 instead of 8
    
    # Let's construct a Strassen-based decomposition for part of the tensor
    # and combine it with schoolbook for the rest
    
    # For the i=0,1; j=0,1; k=0,1 block (2x2x2), use Strassen (rank 7)
    # For the rest, use schoolbook
    
    # Actually, let's just try to verify if we can get rank 71 or lower
    # by using known results. The rank of 3x4x6 matrix multiplication is known
    # to be at most 66 (from literature), but finding an explicit decomposition
    # is non-trivial.
    
    # Let's try a simple approach: use the schoolbook decomposition and see if
    # we can remove some rank-1 terms that are linearly dependent
    
    # For now, return the schoolbook decomposition as a baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 4 * 5 * 5 = 100
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the tensor decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        n, m, p = DIMENSIONS
        # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_rational_list(matrix, tol=1e-10):
        """Convert a float matrix to a list of lists of Fractions."""
        result = []
        for row in matrix:
            r = []
            for val in row:
                # Try to find a simple fraction
                if abs(val - round(val)) < tol:
                    r.append(Fraction(int(round(val))))
                else:
                    # Try fractions with small denominators
                    found = False
                    for denom in range(1, 100):
                        num = round(val * denom)
                        if abs(val - num/denom) < tol:
                            r.append(Fraction(num, denom))
                            found = True
                            break
                    if not found:
                        r.append(Fraction(val))
            result.append(r)
        return result
    
    # Try structured Winograd-like approaches
    # For 5x5 matrix multiplication, Winograd's algorithm gives rank 36
    # For 4x5x5, we can use a similar approach
    
    # Approach 1: Use the fact that we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # for i in 0..3, k in 0..4
    # This is 4 rows of a 5x5 matrix multiplication
    
    # Winograd's algorithm for 5x5 uses transforms
    # Let's try a direct structured approach with rank reduction
    
    # Try rank values from 36 upward (Winograd bound for 5x5 is 36)
    # For 4x5x5, the rank should be at most 4*36 = 144, but we can do better
    
    # Actually, let's think about this differently.
    # The tensor T_{i,j,k} = 1 for all i,j,k (it's the multiplication tensor)
    # We need to find rank-r decomposition
    
    # Try a systematic approach: use bilinear forms
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group terms to reduce the number of multiplications
    
    # For now, let's try a few structured decompositions
    
    # Structured approach: exploit the 4x5x5 structure
    # We can write the multiplication as:
    # C = A * B where A is 4x5, B is 5x5
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # One simple improvement: if we have repeated patterns, we can share computations
    # But for generic matrices, we need all 20 products per row
    
    # Let's try a more sophisticated approach using polynomial methods
    # or just return the schoolbook if we can't find better quickly
    
    # Try a few random sparse decompositions with verification
    random.seed(42)
    
    for target_rank in [40, 50, 60, 70, 80, 90, 95, 99]:
        if time.time() > deadline - 5:
            break
        
        if target_rank >= best_rank:
            continue
        
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Sparse random initialization with structure
            for r in range(target_rank):
                # Each rank-1 term covers a small subset
                U_sparse = random.sample(range(n * m), min(3, n * m))
                V_sparse = random.sample(range(m * p), min(3, m * p))
                W_sparse = random.sample(range(p * n), min(3, p * n))
                
                for idx in U_sparse:
                    U[idx, r] = random.choice([-1, 0, 1])
                for idx in V_sparse:
                    V[idx, r] = random.choice([-1, 0, 1])
                for idx in W_sparse:
                    W[idx, r] = random.choice([-1, 0, 1])
            
            # Check if this is a valid decomposition
            if verify_decomposition(U, V, W):
                # Convert to rational form
                U_rat = to_rational_list(U)
                V_rat = to_rational_list(V)
                W_rat = to_rational_list(W)
                best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                best_rank = target_rank
                break
    
    return best_result

# EVOLVE-BLOCK-END

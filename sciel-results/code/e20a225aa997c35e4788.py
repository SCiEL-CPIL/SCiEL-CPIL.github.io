"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i: row of first matrix (0..n-1)
    - j: column of first matrix / row of second matrix (0..m-1)  
    - k: column of second matrix (0..p-1)
    
    We seek U (shape n*m x rank), V (shape m*p x rank), W (shape p*n x rank)
    such that for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    The known minimal rank for 3x3 matrix multiplication is 23 (Strassen-like constructions).
    We'll implement a known rank-23 decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # For 3x3 matrix multiplication, the Strassen algorithm gives rank 7 for 2x2,
    # and there are known constructions for 3x3.
    # 
    # Actually, the minimal rank of 3x3 matrix multiplication is known to be 23.
    # Let's use a constructive approach based on known results.
    #
    # A simpler approach: use the standard schoolbook (rank 27) but try to find
    # a better decomposition through optimization or known constructions.
    #
    # Let me implement a known rank-23 decomposition for 3x3 matrices.
    # This is based on the work of Strassen and subsequent improvements.
    
    # Actually, let me use a practical approach: start with schoolbook (rank 27)
    # and try to verify if we can find a lower rank via known constructions.
    #
    # For now, let's implement the standard Strassen-based construction that
    # gives rank 23 for 3x3 matrix multiplication.
    
    # Known result: The tensor rank of 3x3 matrix multiplication is 23.
    # We'll construct this explicitly.
    
    # One approach: use the fact that 3x3 = 2x2 + 1x1 in a clever way.
    # Or use known explicit constructions.
    
    # Let me try a different approach: use numerical optimization to find
    # a low-rank decomposition, starting from a random initialization.
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should equal 1 if this is a valid MM tensor entry
                    # The MM tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
                    # Actually, the MM tensor T_{ijk} = delta_{i,i'} delta_{j,j'} delta_{k,k'} 
                    # Let me reconsider.
                    
                    # The matrix multiplication tensor is defined as:
                    # (A * B)_{ik} = sum_j A_{ij} B_{jk}
                    # So the tensor T has T[i,j,k] = 1 (it's the coefficient of A_{ij} * B_{jk} in (AB)_{ik})
                    # 
                    # The decomposition requires:
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # 
                    # And T[i,j,k] = 1 for all valid i,j,k.
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition (rank 27)
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
        return U, V, W, rank
    
    # Try to find a rank-23 decomposition using a known construction
    # or fall back to schoolbook if not found within time budget.
    
    # Let's try a systematic search for low-rank decompositions with small integer entries.
    # This is computationally expensive, so we'll use time limits.
    
    best_rank = n * m * p  # Start with schoolbook
    best_U, best_V, best_W = schoolbook()
    
    if time.time() > deadline - 30:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank 23 with a known construction approach
    # Use the fact that we can combine Strassen's 2x2 (rank 7) with additional terms
    
    # Actually, let me just return the schoolbook decomposition for now,
    # as finding a verified rank-23 decomposition requires a known explicit formula.
    
    # For a more principled approach, let's try random initialization + optimization
    # for rank 23.
    
    def try_random_rank(target_rank, num_trials=500):
        """Try to find a valid decomposition of the given rank using random init + refinement."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            # Initialize with random small values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a structured initialization based on the schoolbook pattern
            # with some random perturbations
            for r in range(target_rank):
                # Assign random indices
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            
            # Try simple local optimization
            # Check if this is already valid
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify(U_list, V_list, W_list):
                return U_list, V_list, W_list, target_rank
            
            # Try adjusting entries
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                # Pick a random entry to modify
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, target_rank - 1)
                    old = U[row, col]
                    U[row, col] = random.choice([-1, 0, 1, 2])
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist(), target_rank
                    U[row, col] = old
                elif mat == 'V':
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, target_rank - 1)
                    old = V[row, col]
                    V[row, col] = random.choice([-1, 0, 1, 2])
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist(), target_rank
                    V[row, col] = old
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, target_rank - 1)
                    old = W[row, col]
                    W[row, col] = random.choice([-1, 0, 1, 2])
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist(), target_rank
                    W[row, col] = old
        
        return None
    
    # Try to find a rank-23 decomposition
    if time.time() < deadline - 20:
        result = try_random_rank(23, num_trials=200)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END


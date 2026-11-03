"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition using a structured approach
    # For 4x4 * 4x5, we can try to exploit block structure.
    
    # Let's try a simple block decomposition:
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x5) into 2x5 blocks: B1, B2
    # C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    # This is just schoolbook split into blocks, no rank reduction.
    
    # Let's try a more aggressive approach: use a randomized search for integer/half-integer
    # solutions with rank < 80.
    
    # Given the complexity, let's try a specific known structure.
    # For rectangular matrix multiplication, one approach is to use the "border rank" 
    # or known algorithms.
    
    # Actually, let's try to implement a simple local search for rank 60-70 with 
    # entries in {-1, 0, 1}.
    
    # This is very hard to do efficiently. Let's instead try a different strategy:
    # Use the fact that we can have half-integers, and try to find a decomposition
    # with rank close to the theoretical minimum.
    
    # For now, let's try a few specific ranks with a simple heuristic.
    
    # Try rank 72, 64, 56, 48 with random integer initialization and check.
    # This is unlikely to work, but let's try.
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        # Verify shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check the tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i, j, k.
        # Also, we need to ensure that the decomposition is "exact" in the sense that
        # the bilinear map is correctly represented.
        
        # For each (i, j, k), the sum should be 1.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try a few random decompositions with small integer entries
    np.random.seed(42)
    
    for target_rank in [64, 72, 56, 48]:
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with entries in {-1, 0, 1}
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Convert to list of lists for the return format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END

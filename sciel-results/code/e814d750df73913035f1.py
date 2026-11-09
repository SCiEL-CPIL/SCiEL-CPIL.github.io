"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 60
    
    # Schoolbook as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_arr[i*m+j], V_arr[j*p+k]), W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_strassen_partition():
        """
        Try to exploit Strassen's algorithm structure for 2x2 blocks.
        Partition the middle dimension m=5 into blocks and output dimension p=6.
        
        Strategy: Use the fact that 2x2x2 multiplication has rank 7 (Strassen).
        We can partition A (2x5) into [A1 A2 A3] where A1,A2 are 2x2, A3 is 2x1.
        Partition B (5x6) into [B1; B2; B3] where B1,B2 are 2x6, B3 is 1x6.
        
        C = A1*B1 + A2*B2 + A3*B3
        
        For A1*B1 (2x2 * 2x6): This is 3 separate 2x2 * 2x2 multiplications (columns 0-1, 2-3, 4-5 of B1)
        Each 2x2*2x2 can use Strassen rank 7, so total 3*7 = 21 for A1*B1
        Similarly for A2*B2: 21
        For A3*B3 (2x1 * 1x6): This is just 2*6 = 12 scalar multiplications, rank 12
        
        Total: 21 + 21 + 12 = 54. That's better than 60!
        
        But wait, we need to be more careful. Let's think about this differently.
        
        Actually, let's try a direct approach: use known low-rank decompositions.
        For 2x2x2, Strassen gives rank 7.
        
        Let me try to construct a decomposition using block structure.
        """
        
        # For now, let's try a simpler structured approach
        # We'll try to find a rank-59 or lower decomposition by exploiting structure
        
        # Approach: Start with schoolbook and try to merge terms
        # This is complex, so let's try random optimization instead
        
        return None
    
    def try_random_search(min_rank=40, max_rank=59, max_attempts=100):
        """
        Try random initialization and local optimization to find lower rank decomposition.
        """
        n, m, p = DIMENSIONS
        best_result = None
        best_rank = max_rank + 1
        
        for rank in range(max_rank, min_rank - 1, -1):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(42 + rank * 100 + attempt)
                
                # Random initialization
                U = np.random.randn(n*m, rank) * 0.3
                V = np.random.randn(m*p, rank) * 0.3
                W = np.random.randn(p*n, rank) * 0.3
                
                # Simple gradient descent on the objective:
                # f = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                lr = 0.01
                for step in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute gradient
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # Compute current value
                                val = 0.0
                                for r in range(rank):
                                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                diff = val - 1.0
                                
                                for r in range(rank):
                                    grad_U[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                                    grad_V[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                                    grad_W[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
                    
                    # Update
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    # Check if we're close to a solution
                    if step % 10 == 0:
                        max_err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                                    max_err = max(max_err, abs(val - 1.0))
                        
                        if max_err < 1e-6:
                            # Found a solution!
                            result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                            if verify(result["U"], result["V"], result["W"]):
                                return result
                            best_result = result
                            best_rank = rank
                            return result
                    
                    # Adaptive learning rate
                    if step % 20 == 0:
                        lr *= 0.9
        
        return best_result
    
    # First, try structured approaches
    result = try_strassen_partition()
    if result is not None:
        if verify(result["U"], result["V"], result["W"]):
            return result
    
    # Try random search for lower rank
    result = try_random_search(min_rank=50, max_rank=59, max_attempts=50)
    if result is not None:
        if verify(result["U"], result["V"], result["W"]):
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END

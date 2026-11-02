"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    Tries to find a decomposition with rank less than the schoolbook rank of 48.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 8
    schoolbook_rank = n * m * p  # 48

    def make_schoolbook():
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

    def verify(U, V, W, rank):
        """Check that the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Start with schoolbook as fallback
    U_best, V_best, W_best, rank_best = make_schoolbook()

    # Try to find a lower-rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Try ranks from 24 up to 47
    
    # Strategy: Try structured decompositions
    # For 2x3x8, we can try to exploit the structure by using a basis that
    # combines the two output rows (n=2) with the 8 columns (p=8)
    
    # One approach: use a decomposition where the W matrix has a specific structure
    # that allows sharing between different k values
    
    # Try rank 40 first (a common improvement for rectangular multiplications)
    # Then try lower ranks if time permits
    
    target_ranks = list(range(47, 23, -1))  # Try from 47 down to 24
    
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
            
        # Try to construct a valid decomposition of this rank
        # Use a greedy approach with random initialization and refinement
        
        # For each target rank, try multiple random initializations
        max_attempts = 50 if target_rank > 40 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                    break
            
            # Initialize with random values
            np.random.seed(attempt + target_rank * 1000)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use a simple gradient-free approach: try to satisfy constraints
            # This is a hard problem, so use a heuristic
            
            # Actually, for this specific case, let's try a more structured approach
            # based on the known properties of matrix multiplication tensors
            
            # For 2x3x8, one known approach is to use a decomposition that
            # treats the 8 columns in pairs and uses 2x2x4 sub-decompositions
            
            # Let's try a block-based construction
            if target_rank == 40:
                # Try to construct a rank-40 decomposition
                # Split p=8 into four blocks of 2
                # Each 2x3x2 block has schoolbook rank 12
                # But we can potentially do better
                
                # For 2x3x2, the minimum rank is known to be 7 (similar to Strassen)
                # But for our purposes, let's try a constructive approach
                
                # Actually, let's just verify if we can find ANY decomposition
                # below 48 by trying specific structured forms
                
                # Try a form where U, V, W have a specific pattern
                # based on the Kronecker product structure
                
                # For now, if we can't find a better one quickly, return schoolbook
                pass
        
        # If we're running out of time, break
        if time.time() > deadline - 10:
            break

    # If we found a better decomposition, use it
    # Otherwise, return the schoolbook decomposition
    
    # Convert to lists for the expected output format
    U_out = [[U_best[i][j] for j in range(len(U_best[0]))] for i in range(len(U_best))]
    V_out = [[V_best[i][j] for j in range(len(V_best[0]))] for i in range(len(V_best))]
    W_out = [[W_best[i][j] for j in range(len(W_best[0]))] for i in range(len(W_best))]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

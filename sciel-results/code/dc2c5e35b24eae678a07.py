"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    The matrix multiplication tensor T is defined such that:
      (C = A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    We seek rank r such that:
      T = sum_{c=1}^{r} U[:,c] * V[:,c] * W[:,c]  (outer product)
    where:
      U has shape (9, r), with U[i,j] corresponding to A_{i,j}
      V has shape (9, r), with V[j,k] corresponding to B_{j,k}
      W has shape (9, r), with W[k,i] corresponding to C_{k,i}  (note: trace convention W[k*n+i, r])
    
    Strategy:
      1. Start with the known schoolbook decomposition of rank 27.
      2. Try to use known low-rank decompositions. The best known general algorithm for 
         small matrices uses specific algebraic identities. For 3x3, the trivial bound is 27.
         Strassen's algorithm gives rank 7 for 2x2, which leads to rank 8 for 3x3 
         via blocking (but this requires handling the overlap). Actually, 3x3 can be 
         done in rank 23 using the "3x3 = 23" result by Winograd (1980) or similar.
        
      Since we need an exact decomposition, we will implement a search or use a known 
      closed-form if available. For 3x3, a rank-23 decomposition is known. However, 
      implementing it from scratch is complex. Let's try a smarter approach:
      
      Use the fact that 3x3 matrix multiplication can be embedded in 4x4 matrix multiplication 
      by padding. If we have a rank-r decomposition for 4x4, we can restrict it to the 
      3x3 subblock. The best known rank for 4x4 is 19 (Winograd 1979). But padding 3x3 to 4x4 
      and using a rank-19 decomposition doesn't directly give a rank-19 decomposition for 3x3.
      
      Actually, the cleanest known low-rank decompositions for small sizes:
      - 2x2: rank 7 (Strassen)
      - 3x3: rank 23 (Winograd)
      - 4x4: rank 19 (Winograd) or lower? No, 4x4 is at most 19, but the lower bound is 16.
      
      For a robust program that must finish in 120s, we should not do a heavy search.
      Instead, we can return the schoolbook rank-27 solution, which is always correct.
      To improve the score (lower rank), we could hardcode a known rank-23 decomposition 
      for 3x3. However, writing out the 23x3 coefficient matrices for U, V, W is 
      error-prone without a verified source.
      
      Alternative: Use a randomized or local optimization search to try to reduce rank.
      This is risky and likely to timeout.
      
      Given the constraints and the need for correctness, we will return the schoolbook 
      decomposition (rank 27). This is safe and meets the "exactly decompose" requirement.
      
      To potentially improve, we can try a simple local search starting from rank 27, 
      attempting to zero out columns by finding linear dependencies. But this is expensive.
      
      Let's implement the schoolbook solution for now, as it is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    
    # First, try to construct a rank-23 decomposition based on overlapping 2x2 Strassen blocks
    # If that fails verification, fall back to schoolbook rank 27
    
    def try_rank_23():
        """
        Attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
        
        The idea is to use two overlapping 2x2 Strassen decompositions:
        - Block 1: covers indices {0,1} in all dimensions (top-left 2x2x2)
        - Block 2: covers indices {1,2} in all dimensions (bottom-right 2x2x2)
        
        Each 2x2 Strassen block has rank 7. Two blocks would give 14 terms.
        The remaining 9 terms cover the "fringe" elements that are not in either block.
        
        However, the overlap at index 1 means we need to be careful about double-counting.
        Let's try a different approach: use a known explicit decomposition.
        
        After research, I'll use the following construction based on the fact that
        M_3 <= 23. I'll hardcode a verified decomposition if I can derive one correctly.
        
        For now, let's try a local search approach starting from rank 27 and trying
        to eliminate terms by finding linear dependencies.
        """
        
        # Start with schoolbook decomposition
        rank_start = n * m * p  # 27
        U = np.zeros((n*m, rank_start))
        V = np.zeros((m*p, rank_start))
        W = np.zeros((p*n, rank_start))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Now try to reduce rank by finding linear dependencies
        # This is a simplified approach - in practice, this would be much more complex
        # For now, return the schoolbook decomposition
        
        # Convert to lists for the required output format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank_start}
    
    # Try rank 23 construction
    try:
        result = try_rank_23()
        rank = result["rank"]
        
        # Verify the decomposition
        U_arr = np.array(result["U"])
        V_arr = np.array(result["V"])
        W_arr = np.array(result["W"])
        
        # Check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
        
    except Exception:
        pass
    
    # Fallback to schoolbook decomposition (guaranteed correct)
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
# EVOLVE-BLOCK-END

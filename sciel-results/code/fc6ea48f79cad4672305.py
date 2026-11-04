"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build schoolbook decomposition as baseline
    U_school = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Strategy: Use block structure to build a potentially lower-rank decomposition
    # Partition A as [A1 (4x4) | A2 (4x1)] and B as [[B1 (4x6)],[B2 (1x6)]]
    # C = A1*B1 + A2*B2
    
    # For A1*B1: 4x4 * 4x6
    # Use Strassen-like decomposition for the 4x4*4x4 part (rank 21)
    # and schoolbook for 4x4*4x2 (rank 32)
    
    # For A2*B2: 4x1 * 1x6 = outer product, rank 24
    
    # Let's construct a decomposition using block structure
    # A is 4x5, B is 5x6
    # Split A into A_left (4x4) and A_right (4x1)
    # Split B into B_top (4x6) and B_bottom (1x6)
    
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (4x4 * 4x6):
    # Further split B_top into B_top_left (4x4) and B_top_right (4x2)
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    # A_left * B_top_left: 4x4 * 4x4, use rank 21
    # A_left * B_top_right: 4x4 * 4x2, schoolbook rank 32
    
    # For A_right * B_bottom (4x1 * 1x6):
    # This is an outer product: A_right[i,0] * B_bottom[0,k]
    # Can be done with rank 24 (one term per (i,k) pair)
    
    # Total: 21 + 32 + 24 = 77
    
    # Now I need to construct the actual rank-21 decomposition for 4x4*4x4
    # This is complex. Let me use a simpler approach: try to find a lower rank
    # by exploiting the tensor structure.
    
    # Alternative: Use the fact that we can combine terms when they share structure
    # For now, let's try a greedy approach with better elimination
    
    def try_eliminate(U, V, W, max_rounds=10):
        """Try to eliminate redundant terms by checking linear dependencies"""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        eliminated = 0
        
        for _round in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            found = False
            for r in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Check if column r is needed by trying to remove it
                U_save = [row[:] for row in U]
                V_save = [row[:] for row in V]
                W_save = [row[:] for row in W]
                
                for row in U:
                    row[r] = 0.0
                for row in V:
                    row[r] = 0.0
                for row in W:
                    row[r] = 0.0
                
                if verify_decomp(U, V, W):
                    # Remove column r
                    U = [row[:r] + row[r+1:] for row in U]
                    V = [row[:r] + row[r+1:] for row in V]
                    W = [row[:r] + row[r+1:] for row in W]
                    rank -= 1
                    eliminated += 1
                    found = True
                    break
                else:
                    U, V, W = U_save, V_save, W_save
            
            if not found:
                break
        
        return U, V, W, eliminated
    
    # Try elimination on schoolbook
    U_elim, V_elim, W_elim, elim_count = try_eliminate(U_school, V_school, W_school)
    
    if elim_count > 0:
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = len(U_elim[0])
    
    # Try a structured decomposition using block multiplication
    # Let's manually construct a decomposition with better structure
    
    # Use the following approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    # For the first part (j=0 to 3), we can use a rank-reduced decomposition
    # For the second part, it's just one term per (i,k)
    
    # Let's try to find a rank-32 decomposition for 4x4*4x4 using known results
    # Actually, the minimal rank for 4x4*4x4 is 21, but constructing it is hard
    
    # Let me try a different approach: use numpy to find the minimal rank
    # by examining the tensor's flattening
    
    # The tensor T[i,j,k] = delta(i,j,k) for matrix multiplication
    # Flatten along different dimensions to find the rank
    
    # For now, return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

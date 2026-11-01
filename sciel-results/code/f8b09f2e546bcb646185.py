"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # Based on the induction argument, rank might be around 2p+2 = 32
    # Or even lower. Let's try to construct rank 32 explicitly.
    
    def try_construct_rank_32():
        R = 32
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Initialize with schoolbook for first 2 columns?
        # Actually, let's just use the induction:
        # Start with p=1, rank 4.
        # Add pairs of columns.
        
        # Base case p=1
        # Terms 0-3
        # c=0: A00 B00 -> C00
        # c=1: A01 B10 -> C00
        # c=2: A10 B00 -> C10
        # c=3: A11 B10 -> C10
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # A00 B00 C00
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1 # A01 B10 C00 (W index k*n+i = 0*2+0=0)
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1 # A10 B00 C10 (W index 0*2+1=1)
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1 # A11 B10 C10
        
        # Wait, W index is k*n+i.
        # For k=0, i=0: index 0.
        # For k=0, i=1: index 1.
        
        # Check base case:
        # i=0, j=0, k=0: U[0,0]*V[0,0]*W[0,0] = 1. Correct.
        # i=0, j=1, k=0: U[1,1]*V[1,1]*W[0,1] = 1. Correct.
        # i=1, j=0, k=0: U[2,2]*V[0,2]*W[1,2] = 1. Correct.
        # i=1, j=1, k=0: U[3,3]*V[1,3]*W[1,3] = 1. Correct.
        
        # Now add columns k=1, 2, ..., 14 in pairs.
        # Pair (1,2), (3,4), ..., (13,14).
        # For each pair (k1, k2), add 2 terms.
        
        current_rank = 4
        for pair_idx in range(7):
            k1 = 2 * pair_idx + 1
            k2 = 2 * pair_idx + 2
            
            c1 = current_rank
            c2 = current_rank + 1
            
            # Term c1 handles j=0 for k1 and k2?
            # Let's try:
            # V[0, k1, c1] = 1, V[0, k2, c1] = 1
            # V[1, k1, c1] = 0, V[1, k2, c1] = 0
            # V[0, k1, c2] = 0, V[0, k2, c2] = 0
            # V[1, k1, c2] = 1, V[1, k2, c2] = 1
            
            # Then for k1:
            # i=0, j=0: U[0, c1] * 1 * W[k1*2+0, c1] = 1 => U[0,c1]=1, W[k1*2, c1]=1
            # i=0, j=1: U[1, c2] * 1 * W[k1*2+0, c2] = 1 => U[1,c2]=1, W[k1*2, c2]=1
            # i=1, j=0: U[2, c1] * 1 * W[k1*2+1, c1] = 1 => U[2,c1]=1, W[k1*2+1, c1]=1
            # i=1, j=1: U[3, c2] * 1 * W[k1*2+1, c2] = 1 => U[3,c2]=1, W[k1*2+1, c2]=1
            
            # For k2:
            # i=0, j=0: U[0, c1] * 1 * W[k2*2+0, c1] = 1 => W[k2*2, c1]=1
            # i=0, j=1: U[1, c2] * 1 * W[k2*2+0, c2] = 1 => W[k2*2, c2]=1
            # i=1, j=0: U[2, c1] * 1 * W[k2*2+1, c1] = 1 => W[k2*2+1, c1]=1
            # i=1, j=1: U[3, c2] * 1 * W[k2*2+1, c2] = 1 => W[k2*2+1, c2]=1
            
            # Set values
            U[0, c1] = 1
            U[1, c2] = 1
            U[2, c1] = 1
            U[3, c2] = 1
            
            V[0 * p + k1, c1] = 1
            V[0 * p + k2, c1] = 1
            V[1 * p + k1, c2] = 1
            V[1 * p + k2, c2] = 1
            
            W[k1 * n + 0, c1] = 1
            W[k2 * n + 0, c1] = 1
            W[k1 * n + 1, c2] = 1
            W[k2 * n + 1, c2] = 1
            W[k1 * n + 0, c2] = 0 # Wait, U[1,c2] is non-zero, so W[k1*n+0, c2] must be 1
            W[k2 * n + 0, c2] = 1
            W[k1 * n + 1, c1] = 0 # Wait, U[2,c1] is non-zero, so W[k1*n+1, c1] must be 1
            W[k2 * n + 1, c1] = 1
            
            # Correction:
            # i=0, j=1: U[1, c2] * V[1, k, c2] * W[k, 0, c2] = 1
            # V[1, k1, c2] = 1. So W[k1*2+0, c2] = 1.
            # V[1, k2, c2] = 1. So W[k2*2+0, c2] = 1.
            
            # i=1, j=0: U[2, c1] * V[0, k, c1] * W[k, 1, c1] = 1
            # V[0, k1, c1] = 1. So W[k1*2+1, c1] = 1.
            # V[0, k2, c1] = 1. So W[k2*2+1, c1] = 1.
            
            # So:
            W[k1 * n + 0, c1] = 1
            W[k2 * n + 0, c1] = 1
            W[k1 * n + 0, c2] = 1
            W[k2 * n + 0, c2] = 1
            W[k1 * n + 1, c1] = 1
            W[k2 * n + 1, c1] = 1
            W[k1 * n + 1, c2] = 1
            W[k2 * n + 1, c2] = 1
            
            # Wait, this sets W to 1 for both c1 and c2 for all i,k.
            # Let's check i=0, j=0, k=k1:
            # Sum_c U[0,c] V[0,k1,c] W[k1,0,c]
            # c=c1: U[0,c1]=1, V[0,k1,c1]=1, W[k1,0,c1]=1 => 1
            # c=c2: U[0,c2]=0, V[0,k1,c2]=0, W[k1,0,c2]=1 => 0
            # Total 1. Correct.
            
            # i=0, j=1, k=k1:
            # c=c1: U[1,c1]=0, V[1,k1,c1]=0, W[k1,0,c1]=1 => 0
            # c=c2: U[1,c2]=1, V[1,k1,c2]=1, W[k1,0,c2]=1 => 1
            # Total 1. Correct.
            
            # i=1, j=0, k=k1:
            # c=c1: U[2,c1]=1, V[0,k1,c1]=1, W[k1,1,c1]=1 => 1
            # c=c2: U[2,c2]=0, V[0,k1,c2]=0, W[k1,1,c2]=1 => 0
            # Total 1. Correct.
            
            # i=1, j=1, k=k1:
            # c=c1: U[3,c1]=0, V[1,k1,c1]=0, W[k1,1,c1]=1 => 0
            # c=c2: U[3,c2]=1, V[1,k1,c2]=1, W[k1,1,c2]=1 => 1
            # Total 1. Correct.
            
            current_rank += 2
        
        # Verify
        # Reconstruct tensor
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T_reconstructed[i,j,k] = val
                    if abs(val - 1) > 1e-9:
                        return None
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result_32 = try_construct_rank_32()
    if result_32:
        return result_32
        
    return best_result
# EVOLVE-BLOCK-END

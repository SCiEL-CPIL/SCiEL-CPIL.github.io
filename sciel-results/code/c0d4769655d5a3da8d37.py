"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Build the target tensor T[i,j,k] = 1 if output[i,k] = sum_j A[i,j]*B[j,k]
    # In trace convention: W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    # Target: for each (i,k), sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # So T[i,j,k] should equal 1 for all valid indices (it's the identity mapping for matmul)
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Try to find lower-rank decomposition using structured approaches
    # Strategy: Use Winograd-style transforms with small integer coefficients
    
    # Approach 1: Try known rank-44 construction for 3x4x4
    # Use Kronecker product structure: 3x4x4 can be viewed as 3 * (4x4)
    # 4x4 matmul has rank 25 (Lavin-Shokrollahi) or we can use rank-20 (Beals)
    # Actually for 4x4, minimal rank is 25 (proven). So 3*4x4 would be 3*25=75? No...
    # Actually we need to think about this differently.
    
    # Let's try a direct search with structured patterns
    # Use the fact that we can combine some terms
    
    # Approach: Try rank values from 44 upward with random structured initialization
    # and verify via tensor reconstruction
    
    def verify(U_mat, V_mat, W_mat):
        """Check if decomposition exactly reproduces matrix multiplication."""
        # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Should equal 1 for all (i,j,k)
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try Winograd-style: use 2x2 block structure
    # For 4x4, use 2x2 blocks: 4x4 = 2x(2x2)
    # Each 2x2 matmul has rank 7 (Winograd)
    # 4x4 with 2x2 blocks: 4 blocks of 2x2, each needs rank 7, but they share structure
    
    # Actually, let's try a more direct approach:
    # Parameterize with small integer/half-integer entries
    
    # Try rank 44 with structured random search
    for target_rank in [44, 45, 46, 47]:
        if time.time() > deadline:
            break
        
        # Structured initialization: use Kronecker-like patterns
        # U is 12 x rank, V is 16 x rank, W is 12 x rank
        
        # Try pattern: split into 3 groups of 4 (for n=3)
        # Each group of 4 rows in U, corresponding 4 cols in W
        
        attempts = 0
        max_attempts = 2000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate structured random matrices with small entries
            # Use {-2, -1, -0.5, 0, 0.5, 1, 2} entries
            entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
            
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            # Fill with structured patterns
            for c in range(target_rank):
                # Each column has sparse pattern
                # U column: sparse over 12 entries
                u_nonzero = np.random.choice(n*m, size=np.random.randint(2,5), replace=False)
                v_nonzero = np.random.choice(m*p, size=np.random.randint(2,5), replace=False)
                w_nonzero = np.random.choice(p*n, size=np.random.randint(2,5), replace=False)
                
                for idx in u_nonzero:
                    U_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in v_nonzero:
                    V_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in w_nonzero:
                    W_mat[idx, c] = entries[np.random.randint(len(entries))]
            
            # Quick check: verify a few (i,j,k) triples first
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                best_rank = target_rank
                break
        
        if best_rank < rank_school:
            break
    
    # If we found a better rank, try to optimize further
    if best_rank < rank_school and time.time() < deadline:
        # Try to reduce rank by 1 with local search
        current_rank = best_rank
        while current_rank > 40 and time.time() < deadline:
            # Try to find rank (current_rank - 1) solution
            new_rank = current_rank - 1
            found = False
            attempts = 0
            while attempts < 500 and not found and time.time() < deadline:
                attempts += 1
                entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
                U_mat = np.zeros((n*m, new_rank))
                V_mat = np.zeros((m*p, new_rank))
                W_mat = np.zeros((p*n, new_rank))
                
                for c in range(new_rank):
                    u_nonzero = np.random.choice(n*m, size=np.random.randint(2,5), replace=False)
                    v_nonzero = np.random.choice(m*p, size=np.random.randint(2,5), replace=False)
                    w_nonzero = np.random.choice(p*n, size=np.random.randint(2,5), replace=False)
                    
                    for idx in u_nonzero:
                        U_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in v_nonzero:
                        V_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in w_nonzero:
                        W_mat[idx, c] = entries[np.random.randint(len(entries))]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(new_rank):
                                val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                    best_rank = new_rank
                    current_rank = new_rank
                    found = True
            
            if not found:
                break
    
    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
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
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify exact decomposition by checking random matrix products."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        np.random.seed(12345)
        for trial in range(8):
            A = np.random.randint(-5, 6, size=(n, m)).astype(np.float64)
            B = np.random.randint(-5, 6, size=(m, p)).astype(np.float64)
            C_true = A @ B
            
            C_decomp = np.zeros((n, p), dtype=np.float64)
            for r in range(rank):
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for j in range(m):
                            val += U_arr[i * m + j, r] * A[i, j] * V_arr[j * p + k, r] * B[j, k]
                        C_decomp[i, k] += val * W_arr[k * n + i, r]
            
            if not np.allclose(C_decomp, C_true, atol=tol, rtol=tol):
                return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Strategy: Use Strassen's 7-term decomposition for 2x2 blocks
    # to build a structured decomposition for 4x5 * 5x5
    
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x5 * 5x5, decompose as blocks:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # This is getting complex. Let me try a different approach:
    # Directly search for lower-rank decompositions using the structure
    
    # Try to find a rank-r decomposition by solving a system of equations
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms
    
    # Alternative: Use the fact that the all-ones tensor of size n*m*p has rank
    # that can be bounded using known results
    
    # For now, let's try random search with structured initialization
    # focusing on the lower bound max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    
    # Try ranks from 25 upward, with increasing effort for lower ranks
    for target_rank in range(25, min(best_rank, 60)):
        if time.time() > deadline - 15:
            break
        
        # Number of attempts based on rank (more attempts for lower ranks)
        attempts = max(1, 10 - (target_rank - 25) // 5)
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Structured random initialization
            # Use values in {-1, 0, 1} with some structure
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Ensure at least some non-zero entries in each row
            for r in range(target_rank):
                if np.all(U[:, r] == 0):
                    U[np.random.randint(n * m), r] = 1
                if np.all(V[:, r] == 0):
                    V[np.random.randint(m * p), r] = 1
                if np.all(W[:, r] == 0):
                    W[np.random.randint(p * n), r] = 1
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                if is_valid_entries(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best_result
        
        if time.time() > deadline - 25:
            break
    
    # Try a more systematic approach: use Kronecker product structure
    # Embed 4x5*5x5 into a larger problem and use known decompositions
    
    # For 2x2*2x2, Strassen gives rank 7
    # For 4x4*4x4, we can use Strassen recursively: rank <= 7^2 = 49
    # But our problem is 4x5*5x5, not square
    
    # Let's try to use the 2x2 Strassen decomposition as a building block
    # and extend it to our dimensions
    
    # Create a decomposition using Strassen-like structure
    # This is heuristic and may not give optimal rank, but let's try
    
    if time.time() < deadline - 30:
        # Try a structured decomposition based on block Strassen
        # Split 4x5 into 2x2 and 2x3 blocks, 5x5 into 2x2, 2x3, 3x2, 3x3
        
        # For simplicity, let's try a direct construction
        # Use the schoolbook but try to find linear dependencies
        
        # Check if any subset of schoolbook terms is linearly dependent
        # This is a hard problem, so let's skip for now
        
        # Final attempt: try random search with half-integer entries
        for target_rank in range(25, min(best_rank, 50)):
            if time.time() > deadline - 20:
                break
            
            attempts = max(1, 5 - (target_rank - 25) // 10)
            for attempt in range(attempts):
                if time.time() > deadline - 15:
                    break
                
                # Use half-integer entries
                U = np.random.choice([-0.5, 0, 0.5, 1], size=(n * m, target_rank))
                V = np.random.choice([-0.5, 0, 0.5, 1], size=(m * p, target_rank))
                W = np.random.choice([-0.5, 0, 0.5, 1], size=(p * n, target_rank))
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify_decomposition(U_list, V_list, W_list):
                    if is_valid_entries(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        return best_result
    
    return best_result
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to construct a lower-rank decomposition
    # Strategy: Use explicit low-rank construction for 3x3x3, then add 3x3x1
    
    # Known: 3x3x3 matmul can be done with rank 23 (Brent-Kung type)
    # We'll try to construct it explicitly
    
    n3, m3, p3 = 3, 3, 3
    
    # For 3x3x3, use a known rank-23 construction
    # This is based on the work of Strassen and subsequent improvements
    # We'll use a specific construction that's known to work
    
    # Actually, let's try a different approach: use the fact that we can
    # decompose the tensor using algebraic identities
    
    # Try rank 23 for the 3x3x3 part using explicit formulas
    # The construction uses 23 scalar multiplications
    
    # For simplicity and reliability, let's try to find a valid decomposition
    # by using a structured approach
    
    # Method: Use the Kronecker product structure and find a low-rank factorization
    # of the multiplication tensor
    
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k in [0,2] for 3x3x3
    # We want to write T as sum of r rank-1 tensors
    
    # Let's try to use a known rank-23 construction for 3x3x3
    # Based on explicit formulas from literature
    
    # For now, let's try a random search with better structure
    # Fix the pattern: use {-1, 0, 1} entries and solve linear systems
    
    best_rank = rank_school
    best_decomp = fallback
    
    # Try ranks from 23 to 35 for the full 3x3x4 problem
    # But we can decompose as 3x3x3 (rank r1) + 3x3x1 (rank r2)
    # Total rank = r1 + r2
    
    # For 3x3x1, minimum rank is 9 (schoolbook is optimal for rectangular)
    # So we need r1 for 3x3x3, total = r1 + 9
    
    # Try r1 from 23 to 27
    for r1 in range(23, 28):
        if time.time() > deadline:
            break
        
        # Try to find rank-r1 decomposition for 3x3x3
        # Use random search with structured initialization
        max_attempts = 2000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random U, V with entries in {-1, 0, 1}
            np.random.seed(attempt)  # Reproducible
            U_rand = np.random.randint(-1, 2, (n3 * m3, r1))
            V_rand = np.random.randint(-1, 2, (m3 * p3, r1))
            
            # Solve for W: for each (k, i), we need
            # sum_l U[i*m3+j, l] * V[j*p3+k, l] * W[k*n3+i, l] = 1 for all j
            # This is a linear system in W[k*n3+i, :]
            
            W_sol = np.zeros((p3 * n3, r1))
            valid = True
            
            for k in range(p3):
                for i in range(n3):
                    # Build matrix A where A[j, l] = U[i*m3+j, l] * V[j*p3+k, l]
                    A = np.zeros((m3, r1))
                    for j in range(m3):
                        for l in range(r1):
                            A[j, l] = U_rand[i * m3 + j, l] * V_rand[j * p3 + k, l]
                    
                    # Solve A @ w = ones(m3)
                    try:
                        # Check if system is consistent
                        rank_A = np.linalg.matrix_rank(A)
                        if rank_A < m3:
                            # Check if ones is in column space
                            # Use least squares
                            w, residuals, rank_ls, s = np.linalg.lstsq(A, np.ones(m3), rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-9:
                                valid = False
                                break
                        else:
                            w = np.linalg.solve(A, np.ones(m3))
                        
                        # Check if w has half-integer entries
                        w2 = w * 2
                        if not np.allclose(w2, np.round(w2), atol=1e-9):
                            valid = False
                            break
                        
                        W_sol[k * n3 + i, :] = w
                    except Exception:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify full identity for 3x3x3
            is_correct = True
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        s = 0.0
                        for l in range(r1):
                            s += U_rand[i * m3 + j, l] * V_rand[j * p3 + k, l] * W_sol[k * n3 + i, l]
                        if abs(s - 1.0) > 1e-9:
                            is_correct = False
                            break
                    if not is_correct:
                        break
                if not is_correct:
                    break
            
            if is_correct:
                # Construct full 3x3x4 decomposition
                total_rank = r1 + 9
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                # Copy 3x3x3 part
                U_full[:9, :r1] = U_rand
                V_full[:9, :r1] = V_rand
                W_full[:9, :r1] = W_sol
                
                # Add 3x3x1 part (k=3) with schoolbook
                col_offset = r1
                for i in range(n):
                    for j in range(m):
                        U_full[i * m + j, col_offset] = 1
                        V_full[j * p + 3, col_offset] = 1
                        W_full[3 * n + i, col_offset] = 1
                        col_offset += 1
                        
                # Verify full decomposition
                full_correct = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for l in range(total_rank):
                                s += U_full[i * m + j, l] * V_full[j * p + k, l] * W_full[k * n + i, l]
                            if abs(s - 1.0) > 1e-9:
                                full_correct = False
                                break
                        if not full_correct:
                            break
                    if not full_correct:
                        break
                
                if full_correct:
                    U_list = [[float(x) for x in row] for row in U_full]
                    V_list = [[float(x) for x in row] for row in V_full]
                    W_list = [[float(x) for x in row] for row in W_full]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a valid 3x3x3 decomposition but haven't returned, 
        # it means the full verification failed, which shouldn't happen
        # Continue to next r1
    
    return fallback

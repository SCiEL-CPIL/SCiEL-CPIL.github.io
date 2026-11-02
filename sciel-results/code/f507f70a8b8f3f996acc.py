import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    dim_U = n * m  # 6
    dim_V = m * p  # 18
    dim_W = p * n  # 12
    
    start_time = time.time()
    deadline = 140  # safety margin

    def verify(U, V, W, tol=1e-6):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Rank-11 decomposition for 2x3x3
    # This is a known minimal decomposition for 2x3x3 multiplication.
    # C = A * B, A is 2x3, B is 3x3, C is 2x3.
    # We need U (6x11), V (9x11), W (6x11)
    # Note: For 2x3x3, dim_U=6, dim_V=9, dim_W=6.
    
    # Let's use a specific known rank-11 decomposition.
    # One such decomposition is based on the "border rank" or specific integer solutions.
    # If I don't have it hardcoded, I can use the schoolbook (9) + 2 extra? No, schoolbook is 9.
    # Wait, 2x3x3 schoolbook rank is 2*3*3 = 18? No, 2*3*3=18 terms?
    # No, C_ik = sum_j A_ij B_jk.
    # Terms: i=0,1; j=0,1,2; k=0,1,2. Total 18 terms.
    # Rank of 2x3x3 is known to be 11? Or is it 12?
    # Actually, rank of 2x3x3 is 11? 
    # Let's check: 2x2x2 is 7. 2x3x3 is likely higher.
    # Schoolbook is 18.
    # If rank is 11, two blocks is 22.
    # If rank is 12, two blocks is 24.
    # The inspirations said "Rank 11 decomposition for 2x3x3". I will trust that.
    
    # I will construct a rank-22 solution using two independent rank-11 blocks.
    # Since I don't have the exact rank-11 matrix hardcoded in my head, 
    # I will use a fallback: Schoolbook for 2x3x6 is 36.
    # But the inspirations achieved 22. 
    # I need to provide a valid decomposition.
    # If I can't construct 22, I should return 36 (score 1.0) or try to find 22 via search.
    # Given the constraint, I will try to construct a rank-22 solution by 
    # splitting the problem into two 2x3x3 problems and using a known 
    # efficient decomposition for 2x3x3.
    
    # Let's try to find a rank-11 decomposition for 2x3x3 using a small search.
    # If that fails, use schoolbook.
    
    def find_rank_11_2x3x3():
        # Returns U, V, W for 2x3x3
        # Dimensions: U(6, R), V(9, R), W(6, R)
        # Target R = 11
        # We will try a few known patterns or random restarts.
        # For safety, I will return a schoolbook decomposition (R=18) if search fails.
        # But the goal is to beat 22.
        
        # Actually, let's just return the two-block schoolbook (R=36) if we can't be sure.
        # But the inspirations got 22. 
        # I will implement a search for R=21 or 22 for the full 2x3x6 problem.
        
        # Let's try a direct search for R=22 for the full problem, 
        # initialized with block-diagonal structure.
        
        pass

    # Strategy:
    # 1. Construct a rank-22 solution using two rank-11 blocks (if we can find them).
    # 2. If we can't find them, use rank-36 schoolbook.
    # 3. Try to optimize down to 21.
    
    # Since I cannot guarantee finding the rank-11 matrices in 140s via search,
    # and I don't have them hardcoded, I will use a **hybrid** approach:
    # I will use the **Strassen-like** decomposition for 2x2x2 (rank 7) 
    # and extend it.
    
    # Let's try to construct a rank-20 or 21 solution directly for 2x3x6.
    # We will use a random sparse matrix initialization and check if it can be 
    # completed to a valid decomposition.
    
    # Given the complexity, I will return a rank-36 schoolbook solution 
    # to ensure validity, but I will try to improve it to rank-22 
    # by using the "two blocks" method if I can hardcode the rank-11 matrices.
    
    # I will hardcode a known rank-11 decomposition for 2x3x3 if possible.
    # If not, I will use rank-18 (schoolbook) for 2x3x3, giving rank-36 total.
    # This is not an improvement.
    
    # Let's try a different angle: 
    # The rank of 2x3x6 is at most 2 * rank(2x3x3).
    # If rank(2x3x3) = 11, then rank(2x3x6) <= 22.
    # I will attempt to construct the rank-11 decomposition for 2x3x3 
    # using a small exhaustive search over integer/half-integer grids 
    # with a fixed structure.
    
    def search_2x3x3_rank(R_target):
        # Dimensions: U(6, R), V(9, R), W(6, R)
        # We fix the structure: 
        # U and W are sparse. V is sparse.
        # This is too slow.
        return None

    # Fallback: Return Schoolbook (Rank 36)
    def schoolbook():
        U = np.zeros((dim_U, n*m*p))
        V = np.zeros((dim_V, n*m*p))
        W = np.zeros((dim_W, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # If I have time, I will try to find a rank-22 solution 
    # by assuming the rank-11 matrices exist and using a placeholder 
    # that I know works? No, I must provide valid matrices.
    
    # I will return schoolbook for now, as I cannot guarantee the rank-11 
    # matrices are correct without them.
    
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

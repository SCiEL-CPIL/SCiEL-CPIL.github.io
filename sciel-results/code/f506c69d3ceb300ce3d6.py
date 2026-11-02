import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    # Target tensor structure
    # T[i, j, k] = 1 if j matches, 0 otherwise.
    # Indices:
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i  (Note: prompt says W[k*n+i, r])
    
    # We want to find U, V, W of rank R such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j is the "matching" index?
    # Wait, the tensor T is defined by the multiplication.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i, j] B[j, k] contributes to C[i, k].
    # So for the specific triple (i, j, k), the coefficient is 1.
    # For a triple (i, j, k') where k' != k? No, k is part of the input B.
    # The tensor T has indices (alpha, beta, gamma).
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i  (Wait, gamma is output index. Output is C[i,k]. 
    #               Prompt says W index is k*n+i. So gamma = k*n+i.
    #               This implies output index is (k, i)? 
    #               Standard C is n x p. C[i, k]. 
    #               If W index is k*n + i, then the output is indexed by (k, i).
    #               This means C is stored as C[k, i]? Or the convention is different.
    #               "output factor follows the official trace convention W[k*n+i,r]"
    #               Usually trace convention implies C[i,k] corresponds to index i*p + k.
    #               But here it is k*n + i.
    #               Let's assume the output index is (k, i) in the W matrix.
    #               So the equation is:
    #               sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 
    #               for all i, j, k?
    #               No, only if the term A[i,j]B[j,k] is part of the product.
    #               In standard multiplication, A[i,j]B[j,k] IS part of C[i,k].
    #               So for all i, j, k, the coefficient is 1.
    #               What about terms where j doesn't match?
    #               A[i, j] and B[j', k] with j != j'. 
    #               Do they contribute to C[i, k]? No.
    #               So T[alpha, beta, gamma] = 0 if j_alpha != j_beta.
    
    # So we need:
    # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # For all i, j, j', k (j != j'): sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = 0
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 25 up to best_rank - 1
    for R in range(25, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 # Adjust based on time
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Initialize with random integers
            # Scale to avoid overflow and keep numbers small
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            # Check if solution is valid
            # We can compute the tensor T_calc and compare with T_target
            # T_calc[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
            
            # Efficient check:
            # Iterate over all valid (i, j, k) and invalid (i, j, j', k)
            
            valid = True
            # Check valid terms
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        alpha = i * m + j
                        beta = j * p + k
                        gamma = k * n + i
                        val = np.dot(U[alpha], V[beta]) # Wait, sum_r U[alpha,r] V[beta,r] is not enough.
                        # It is sum_r U[alpha,r] V[beta,r] W[gamma,r]
                        # This is a trilinear form.
                        # We can compute it as:
                        # tmp = U[alpha] * V[beta] (element-wise)
                        # val = np.dot(tmp, W[gamma])
                        tmp = U[alpha] * V[beta]
                        val = np.dot(tmp, W[gamma])
                        if val != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            # Check invalid terms (j != j')
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        if j == jp: continue
                        for k in range(p):
                            alpha = i * m + j
                            beta = jp * p + k
                            gamma = k * n + i
                            tmp = U[alpha] * V[beta]
                            val = np.dot(tmp, W[gamma])
                            if val != 0:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            
            if valid:
                found = True
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break # Found a solution for this R, go to next R (lower is better)
        
        if found:
            # If we found a solution for R, we can try lower R?
            # The loop goes 25, 26... 
            # If we find 25, we break out of the R loop?
            # No, we want the MINIMAL rank.
            # So we should check 25, then 26...
            # If we find 25, that's the best possible (if lower bound is 25).
            # But lower bound is 43.
            # So we will likely not find 25.
            # We should continue to higher R?
            # No, we want to find ANY rank < 50.
            # If we find R=45, score is 50/45.
            # If we find R=44, score is 50/44.
            # So we should try to find the SMALLEST R that works.
            # My loop starts at 25. If it fails, it goes to 26.
            # If it finds 43, it updates best_rank to 43.
            # It should then stop? Or try 42?
            # The loop is `for R in range(25, best_rank)`.
            # If best_rank becomes 43, the range stops at 42.
            # So it will try 25..42.
            # This is correct.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

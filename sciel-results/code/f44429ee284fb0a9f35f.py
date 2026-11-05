import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try
    # Known lower bound is ~12, upper bound 18 (schoolbook).
    # We aim for 14, 15, 16, 17.
    target_ranks = [14, 15, 16, 17]
    
    # Schoolbook fallback (Rank 18)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    np.random.seed(42)
    
    for R in target_ranks:
        # Try multiple random restarts
        # Increase restarts for lower ranks as they are harder
        restarts = 100 if R <= 15 else 50
        
        for restart in range(restarts):
            # Initialize with small random Gaussian noise
            # Scale matters. Let's start with standard normal.
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize columns to unit norm to help convergence
            # This prevents one matrix from dominating the scale
            for r in range(R):
                if np.linalg.norm(U[:, r]) > 1e-10: U[:, r] /= np.linalg.norm(U[:, r])
                if np.linalg.norm(V[:, r]) > 1e-10: V[:, r] /= np.linalg.norm(V[:, r])
                if np.linalg.norm(W[:, r]) > 1e-10: W[:, r] /= np.linalg.norm(W[:, r])
            
            # Alternating Least Squares (ALS) iterations
            for it in range(100):
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # m_ij[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                        m_ij = np.zeros(R)
                        for k in range(p):
                            m_ij += V[j * p + k, :] * W[k * n + i, :]
                        norm_sq = np.dot(m_ij, m_ij)
                        if norm_sq > 1e-12:
                            U_new[i * m + j, :] = m_ij / norm_sq
                        else:
                            U_new[i * m + j, :] = U[i * m + j, :]
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # m_jk[r] = sum_i U[i*m+j, r] * W[k*n+i, r]
                        m_jk = np.zeros(R)
                        for i in range(n):
                            m_jk += U[i * m + j, :] * W[k * n + i, :]
                        norm_sq = np.dot(m_jk, m_jk)
                        if norm_sq > 1e-12:
                            V_new[j * p + k, :] = m_jk / norm_sq
                        else:
                            V_new[j * p + k, :] = V[j * p + k, :]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # m_ki[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                        m_ki = np.zeros(R)
                        for j in range(m):
                            m_ki += U[i * m + j, :] * V[j * p + k, :]
                        norm_sq = np.dot(m_ki, m_ki)
                        if norm_sq > 1e-12:
                            W_new[k * n + i, :] = m_ki / norm_sq
                        else:
                            W_new[k * n + i, :] = W[k * n + i, :]
                W = W_new
            
            # Rational Reconstruction: Round to nearest half-integer
            # The values might be large or small. 
            # We want to find U, V, W such that sum U V W = 1.
            # If we scale U, V, W by factors alpha, beta, gamma with alpha*beta*gamma=1, 
            # the equation holds.
            # The ALS solution finds the minimum norm solution for each step.
            # This tends to keep values small.
            
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            # Verify if the rounded matrices form an exact decomposition
            # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                        if not np.isclose(val, 1.0, atol=1e-9):
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Convert to list of lists for the return interface
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better rank found, return schoolbook
    return fallback
# EVOLVE-BLOCK-END

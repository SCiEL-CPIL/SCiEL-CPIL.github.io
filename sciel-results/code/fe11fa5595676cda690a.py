"""Rewritten program for matrix multiplication 2,2,6 tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    We use a structured approach based on Strassen decomposition for 2x2x2x2
    blocks, then attempt to reduce rank by finding linear dependencies
    among the rank-21 solution that preserve the bilinear identity.
    
    The tensor T[i,j,k] for (i,j,k) in [0,2)x[0,2)x[0,6) equals 1.
    
    We seek U (4 x r), V (12 x r), W (12 x r) such that for all (i,j,k):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Strategy:
    1. Build rank-21 Strassen-based decomposition (3 blocks of 7).
    2. Attempt to reduce to rank 20 by finding a non-trivial linear
       combination of the 21 rank-1 terms that vanishes on the tensor.
    """
    import numpy as np
    
    n, m, p = 2, 2, 6
    nm = n * m  # 4
    mp = m * p  # 12
    pn = p * n  # 12
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2x2():
        """Return U, V, W for 2x2 x 2x2 x 2x2 Strassen decomposition (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        
        # p2 = (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        
        # p3 = a00(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        
        # p4 = a11(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        
        # p5 = (a00+a01)b11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        
        # p6 = (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        
        # p7 = (a01-a11)(b10+b11)
        U[2, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        
        # W[k*n+i] where n=2: W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
        # c00 = p1 + p4 - p5 - p7
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
        # c01 = p3 + p5
        W[1, 2] = 1; W[1, 4] = 1
        # c10 = p2 + p4
        W[2, 1] = 1; W[2, 3] = 1
        # c11 = p1 - p2 + p3 + p6
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W
    
    # Build rank-21 Strassen-based decomposition
    U_strassen, V_strassen, W_strassen = strassen_2x2x2x2()
    
    rank_21 = 21
    U21 = np.zeros((4, rank_21))
    V21 = np.zeros((12, rank_21))
    W21 = np.zeros((12, rank_21))
    
    for t in range(3):
        base = t * 7
        U21[:, base:base+7] = U_strassen
        for j in range(2):
            for k_local in range(2):
                for c in range(7):
                    V21[j * 6 + 2 * t + k_local, base + c] = V_strassen[j * 2 + k_local, c]
        for k_local in range(2):
            for i in range(2):
                for c in range(7):
                    W21[(2 * t + k_local) * 2 + i, base + c] = W_strassen[k_local * 2 + i, c]
    
    # Verify rank-21 decomposition
    if check_decomposition(U21, V21, W21, rank_21):
        # Try to reduce to rank 20 by finding a linear dependency
        # that vanishes on the tensor.
        
        # The idea: if we can find vectors u (4x1), v (12x1), w (12x1) 
        # and coefficients alpha_c such that for all (i,j,k):
        #   sum_c alpha_c * U21[i*m+j, c] * V21[j*p+k, c] * W21[k*n+i, c] = 0
        # then we can subtract this from the decomposition.
        
        # A simpler approach: look for a linear combination of the 21 terms
        # that is identically zero as a function of (A, B).
        
        # For each (i,j,k), the contribution from term c is:
        #   U21[i*m+j, c] * V21[j*p+k, c] * W21[k*n+i, c]
        # We need sum_c alpha_c * (above) = 0 for all (i,j,k).
        
        # This gives us 2*2*6 = 24 linear equations in 21 unknowns (alpha_c).
        # If there's a non-trivial solution, we can use it to reduce rank.
        
        # Set up the system: for each (i,j,k), sum_c alpha_c * f_c(i,j,k) = 0
        # where f_c(i,j,k) = U21[i*m+j, c] * V21[j*p+k, c] * W21[k*n+i, c]
        
        # Build the constraint matrix M (24 x 21)
        M = np.zeros((24, 21))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank_21):
                        M[idx, c] = U21[i * m + j, c] * V21[j * p + k, c] * W21[k * n + i, c]
                    idx += 1
        
        # Find the null space of M
        # If there's a non-trivial null vector, we can reduce the rank.
        try:
            _, S, Vt = np.linalg.svd(M)
            # Check if there are singular values close to zero
            tol = 1e-10
            null_rank = np.sum(S < tol)
            
            if null_rank > 0:
                # There's a non-trivial linear dependency
                # Get the null space vectors
                null_space = Vt[-null_rank:].T  # (21 x null_rank)
                
                # For each null vector, try to eliminate one term
                for ns in range(null_rank):
                    alpha = null_space[:, ns]
                    
                    # Try to eliminate term c where alpha_c is largest
                    max_idx = np.argmax(np.abs(alpha))
                    if abs(alpha[max_idx]) > 1e-10:
                        # Scale so that alpha[max_idx] = 1
                        alpha = alpha / alpha[max_idx]
                        
                        # New rank-20 decomposition:
                        # Remove term max_idx and adjust others
                        # U20 = U21 without column max_idx, but add alpha adjustments
                        # Actually, we need to be more careful.
                        
                        # The new decomposition has terms c != max_idx,
                        # but we need to account for the linear combination.
                        # 
                        # If sum_c alpha_c * T_c = 0, then T_max = -sum_{c != max} (alpha_c/alpha_max) * T_c
                        # So we can express the max_idx term in terms of the others.
                        
                        # Build rank-20 matrices by removing the max_idx column
                        keep = [c for c in range(rank_21) if c != max_idx]
                        
                        U20 = U21[:, keep].copy()
                        V20 = V21[:, keep].copy()
                        W20 = W21[:, keep].copy()
                        
                        # The decomposition now misses the max_idx term,
                        # but since T_max = -sum_{c != max} (alpha_c/alpha_max) * T_c,
                        # we need to add this back. However, the other terms are
                        # already in the decomposition, so we just need to verify.
                        
                        # Actually, the rank-20 decomposition is:
                        # sum_{c in keep} U20[:, c] * V20[:, c] * W20[:, c]
                        # = sum_{c in keep} U21[:, c] * V21[:, c] * W21[:, c]
                        # = (sum_{all c} U21[:, c] * V21[:, c] * W21[:, c]) - U21[:, max_idx] * V21[:, max_idx] * W21[:, max_idx]
                        # = T - T_max
                        # = T - (-sum_{c in keep} (alpha_c/alpha_max) * T_c)
                        # = T + sum_{c in keep} (alpha_c/alpha_max) * T_c
                        # This is NOT equal to T unless the alpha_c are all zero.
                        
                        # So this approach doesn't directly work. We need a different strategy.
                        pass
                
                # Alternative: use the null space to find a lower-rank representation
                # by combining terms.
                
                # For now, if no simple reduction works, return rank 21.
                U_list = U21.tolist()
                V_list = V21.tolist()
                W_list = W21.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # No linear dependency found, return rank 21
                U_list = U21.tolist()
                V_list = V21.tolist()
                W_list = W21.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        except np.linalg.LinAlgError:
            # SVD failed, return rank 21
            U_list = U21.tolist()
            V_list = V21.tolist()
            W_list = W21.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If rank 21 doesn't work (shouldn't happen), fall back to schoolbook
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

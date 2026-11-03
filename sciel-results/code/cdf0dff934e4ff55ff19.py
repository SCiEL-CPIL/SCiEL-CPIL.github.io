"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook rank is n*m*p = 4*4*8 = 128
    # We will try to reduce this rank using a randomized search.
    # Strategy: Start with schoolbook, then try to remove terms one by one.
    # For each term, check if it can be removed by adjusting other terms.
    # This is equivalent to checking if the tensor is in the span of the others.
    
    # The matrix multiplication tensor T has components T[i,j,k] = delta_{i,j,k} (in a specific basis)
    # Actually, T[i,j,k] is 1 if we are computing C[i,k] = sum_j A[i,j] B[j,k]
    # Wait, the tensor is defined by T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # T_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # In the decomposition, U, V, W are such that:
    # T = sum_r U[:,r] V[:,r] W[:,r]^T (as a 3-tensor)
    # Specifically, T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's represent the tensor as a 3D array T[i,j,k]
    # T[i,j,k] = 1 if (i,j,k) is a valid term in the sum for C[i,k].
    # For matrix multiplication, the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} = 1 if the index j is the "middle" index connecting i and k.
    # In standard matrix multiplication, for each output (i,k), we sum over j.
    # The tensor is non-zero at (i,j,k) if and only if the term A[i,j]*B[j,k] is part of the computation.
    # This is true for ALL i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # So the tensor is the "all-ones" tensor? No, that's not right.
    # The tensor T is defined such that <T, A ⊗ B> = C.
    # T_{i,j,k} = 1 for all i,j,k?
    # Let's check: C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
    # If T_{i,j,k} = 1 for all i,j,k, then C_{i,k} = sum_j A_{i,j} B_{j,k}. This is correct.
    # So the matrix multiplication tensor is indeed the all-ones tensor of shape (n, m, p).
    # Wait, is that right?
    # Yes, for standard matrix multiplication, the tensor is the all-ones tensor.
    
    # So we want to decompose the all-ones tensor of shape (4, 4, 8) into a sum of rank-1 tensors.
    # A rank-1 tensor is U_{i,j} V_{j,k} W_{k,i} (with appropriate index permutations).
    # Actually, the decomposition is:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # But the indices must match the "trace convention" W[k*n+i, r].
    # Let's stick to the standard form:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    
    # The schoolbook decomposition is:
    # T_{i,j,k} = sum_{r} delta_{i,j,k}^{(r)} where r=(i,j,k)
    # U_{i,j}^{(r)} = 1 if r=(i,j,k) else 0
    # V_{j,k}^{(r)} = 1 if r=(i,j,k) else 0
    # W_{k,i}^{(r)} = 1 if r=(i,j,k) else 0
    
    # We want to find a decomposition with fewer terms.
    # This is equivalent to finding the tensor rank of the all-ones tensor of shape (4,4,8).
    # The tensor rank of the all-ones tensor of shape (n,m,p) is known to be n*m*p for small dimensions,
    # but can be reduced using algorithms like Strassen.
    
    # For (4,4,8), the schoolbook rank is 128.
    # Can we do better?
    # The tensor rank of the all-ones tensor of shape (n,m,p) is at most n*m*p.
    # It is known that the tensor rank of the all-ones tensor of shape (2,2,2) is 7 (Strassen).
    # For (4,4,4), the tensor rank is at most 49 (using Strassen recursively).
    # For (4,4,8), we can view it as two (4,4,4) problems, so rank at most 98.
    # Or we can try to find a better decomposition.
    
    # Let's use a randomized search to try to find a decomposition with rank < 128.
    # We'll start with rank 128 and try to reduce it.
    
    # Actually, let's just implement the Strassen-based decomposition for (4,4,8).
    # We can split the 8 columns into two groups of 4.
    # C[:, 0:4] = A * B[:, 0:4] and C[:, 4:8] = A * B[:, 4:8].
    # Each is a (4,4,4) matrix multiplication.
    # The tensor rank of (4,4,4) is at most 49 (using Strassen).
    # So total rank is at most 49 + 49 = 98.
    
    # Let's implement the Strassen decomposition for (4,4,4) and use it twice.
    
    def strassen_4x4_4x4():
        """Return U, V, W for 4x4 * 4x4 = 4x4 with rank 49."""
        # This is a known result. The rank of the 4x4x4 matrix multiplication tensor is 49.
        # We can construct it using Strassen's algorithm recursively.
        # Strassen for 2x2x2 has rank 7.
        # For 4x4x4, we split into 2x2 blocks of 2x2 matrices.
        # This gives 7 sub-problems of type (2,4,4)? No.
        # Actually, Strassen's algorithm for 4x4x4 uses 49 multiplications.
        # Let's just hardcode the decomposition or generate it.
        
        # For simplicity, let's use a known decomposition.
        # The rank of the 4x4x4 matrix multiplication tensor is 49.
        # We can find this in the literature or compute it.
        
        # Let's try to compute it using a search.
        # But that might take too long.
        
        # Alternative: Use the schoolbook for (4,4,4) which is 64, and see if we can reduce it.
        # But 64*2 = 128, which is the same as the original.
        
        # Let's just return the schoolbook for now and see if we can improve.
        # Actually, the task says "Find a rank-minimal exact tensor decomposition".
        # The seed program returns rank 128.
        # If we can find a decomposition with rank < 128, we improve the score.
        
        # Let's try to find a decomposition with rank 98 using Strassen.
        # We'll split the 8 columns into two groups of 4.
        # For each group, we use a (4,4,4) decomposition.
        # If we can find a (4,4,4) decomposition with rank 49, total rank is 98.
        
        # Let's assume we can find such a decomposition.
        # For now, let's just return the schoolbook.
        pass
    
    # Let's try a different approach: use a randomized search to find a low-rank decomposition.
    # We'll start with the schoolbook and try to remove terms.
    
    n, m, p = 4, 4, 8
    rank = n * m * p  # 128
    
    # Initialize U, V, W with the schoolbook decomposition
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce the rank by removing terms.
    # We'll try to remove each term and see if the tensor is still exactly decomposed.
    # This is equivalent to checking if the term is in the span of the others.
    
    # For each term r, we check if there exist coefficients c_s for s != r
    # such that the tensor is still exactly decomposed.
    # This is a linear algebra problem.
    
    # Let's try to remove terms one by one.
    # We'll keep a list of active terms.
    active = list(range(rank))
    
    # Try to remove each term
    for r in active[:]:
        if time.time() > _DEADLINE:
            break
        
        # Check if term r can be removed
        # We need to find c_s for s in active \ {r} such that:
        # sum_{s in active \ {r}} c_s * U_s V_s W_s = U_r V_r W_r
        # This is a system of linear equations.
        
        # The tensor equation is:
        # sum_s c_s T_s = T_r
        # where T_s is the rank-1 tensor U_s V_s W_s.
        
        # We can flatten the tensors into vectors and solve a linear system.
        
        # Let's form the matrix A where A[:, s] = vec(T_s) for s in active \ {r}
        # and b = vec(T_r).
        # We want to solve A c = b.
        
        # The size of the tensor is n*m*p = 128.
        # The number of active terms is at most 128.
        # So we're solving a 128x128 linear system, which is fast.
        
        active_without_r = [s for s in active if s != r]
        if not active_without_r:
            continue
        
        # Form the matrix A and vector b
        A = np.zeros((n * m * p, len(active_without_r)))
        for idx, s in enumerate(active_without_r):
            # T_s = U_s V_s W_s
            # T_s[i,j,k] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            T_s = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_s[i,j,k] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            A[:, idx] = T_s.flatten()
        
        # b = vec(T_r)
        T_r = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        b = T_r.flatten()
        
        # Solve A c = b
        try:
            c, residuals, rank_l, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if the solution is exact
            if np.allclose(A @ c, b, atol=1e-10):
                # Term r can be removed
                active.remove(r)
        except:
            pass
    
    # Convert to the required format
    final_rank = len(active)
    U_final = np.zeros((n * m, final_rank))
    V_final = np.zeros((m * p, final_rank))
    W_final = np.zeros((p * n, final_rank))
    
    for new_col, old_col in enumerate(active):
        U_final[:, new_col] = U[:, old_col]
        V_final[:, new_col] = V[:, old_col]
        W_final[:, new_col] = W[:, old_col]
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

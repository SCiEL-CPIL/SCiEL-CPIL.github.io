"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is equivalent to finding the minimum rank of the tensor
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Let's try to use the structure: for fixed i, the terms involve j and k
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (sum_c U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]... no, that's not right
    
    # The correct formulation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_k B[j,k] * V[j*p+k,c]) * W[k*n+i,c]
    # That's not quite right either.
    
    # Let me re-derive:
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] = A[i,j]*B[j,k]
    # for all i,j,k. This means:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # So the tensor T[i,j,k] = 1 for all i,j,k, and we're decomposing it.
    # But wait, that can't be right because the tensor should be sparse.
    
    # I think I'm confusing myself. Let me look at the standard formulation.
    
    # The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ f_{ik}
    # where e_i are basis vectors in C^n, e_j in C^m, f_{ik} = e_i ⊗ e_k in C^n ⊗ C^p
    
    # So M is in (C^n ⊗ C^m) ⊗ (C^m ⊗ C^p) ⊗ (C^n ⊗ C^p)
    # Wait, that's not right. The standard is:
    # M ∈ (C^n ⊗ C^m)^* ⊗ (C^m ⊗ C^p)^* ⊗ (C^n ⊗ C^p)
    # with M = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)
    
    # So the tensor has entries: M[(i,j), (j,k), (i,k)] = 1
    
    # In our indexing:
    # First mode: i*m + j (size n*m = 6)
    # Second mode: j*p + k (size m*p = 27)  
    # Third mode: k*n + i (size p*n = 18)
    
    # So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k
    
    # This is a 3-tensor of size 6 x 27 x 18 with 54 non-zero entries.
    # We want to find the minimum rank decomposition.
    
    # Let's build this tensor and try to find its rank using numerical methods.
    
    tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Now, the rank of a tensor is the minimum r such that:
    # T[a,b,c] = sum_{d=1}^r U[a,d] * V[b,d] * W[c,d]
    
    # This is a hard problem. Let's try a heuristic approach.
    
    # One idea: for each "slice" along one mode, compute the matrix rank.
    # The tensor rank is at least the maximum over all slices of the matrix rank.
    
    # Slice along first mode: for each a, T[a,:,:] is a (m*p) x (p*n) matrix
    slice_ranks_0 = []
    for a in range(n * m):
        mat = tensor[a, :, :]
        # Compute rank
        try:
            r = np.linalg.matrix_rank(mat, tol=1e-10)
        except:
            r = 0
        slice_ranks_0.append(r)
    
    # Slice along second mode: for each b, T[:,b,:] is a (n*m) x (p*n) matrix
    slice_ranks_1 = []
    for b in range(m * p):
        mat = tensor[:, b, :]
        try:
            r = np.linalg.matrix_rank(mat, tol=1e-10)
        except:
            r = 0
        slice_ranks_1.append(r)
    
    # Slice along third mode: for each c, T[:,:,c] is a (n*m) x (m*p) matrix
    slice_ranks_2 = []
    for c in range(p * n):
        mat = tensor[:, :, c]
        try:
            r = np.linalg.matrix_rank(mat, tol=1e-10)
        except:
            r = 0
        slice_ranks_2.append(r)
    
    # The tensor rank is at least max of these
    lower_bound = max(max(slice_ranks_0), max(slice_ranks_1), max(slice_ranks_2))
    
    # For this tensor, let's compute the actual lower bound
    # Each slice T[a,:,:] for fixed a=i*m+j:
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 iff i'=i, j'=j, so k''=k'
    # Wait, let me be more careful.
    
    # T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 iff i'=i and j'=j
    # So for fixed (i,j), T[i*m+j, :, :] has non-zeros at:
    # (j'*p+k', k''*n+i'') where j'=j and i''=i
    # So j'=j, i''=i, and k', k'' are free with k''=k' (since the entry is 1 only when they match? No)
    
    # Actually, T[a,b,c] = 1 where a=i*m+j, b=j'*p+k', c=k''*n+i''
    # and the condition is that there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # So for fixed a=i*m+j: b must be of the form j*p+k for some k, and c must be of the form k'*n+i for some k'
    # And we need k=k' (same k)
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all k
    # This means: for fixed (i,j), the non-zeros are at positions (j*p+k, k*n+i) for k=0,...,p-1
    # This is a matrix with p non-zeros, one in each "diagonal" position
    
    # The rank of this matrix: it's a permutation-like matrix with p ones
    # The rank is min(p, n*m, m*p, p*n) but more precisely, it's the number of linearly independent rows/cols
    
    # For fixed (i,j), the matrix M has M[j*p+k, k*n+i] = 1 for k=0,...,p-1
    # This is a matrix with p ones in distinct rows and distinct columns
    # The rank is p (if p <= min(m*p, p*n), which is true)
    
    # So each slice has rank p = 9
    # Therefore, the tensor rank is at least 9
    
    # But schoolbook gives 54, so there's a gap. Can we achieve rank 9?
    # Probably not, but maybe we can do better than 54.
    
    # Let's try to find a decomposition with rank r for various r
    
    # Use alternating least squares with correct outer product logic
    def try_decomposition(r, num_restarts=5, max_iter=100):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > 10: # 10s budget per restart
                break
            np.random.seed(restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iter):
                if time.time() - start_time > 12: # Hard stop near limit
                    break
                
                # Update U
                # Design matrix columns are vec(V[:,d] * W[:,d]^T)
                design = np.zeros((m * p * p * n, r))
                for d in range(r):
                    # Outer product of V[:,d] (m*p,1) and W[:,d] (1,p*n) -> (m*p, p*n)
                    outer = np.outer(V[:, d], W[:, d])
                    design[:, d] = outer.flatten()
                
                for a in range(n * m):
                    target = tensor[a, :, :].flatten()
                    try:
                        U[a, :], residuals, rank, sv = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Update V
                design = np.zeros((n * m * p * n, r))
                for d in range(r):
                    outer = np.outer(U[:, d], W[:, d])
                    design[:, d] = outer.flatten()
                for b in range(m * p):
                    target = tensor[:, b, :].flatten()
                    try:
                        V[b, :], residuals, rank, sv = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Update W
                design = np.zeros((n * m * m * p, r))
                for d in range(r):
                    outer = np.outer(U[:, d], V[:, d])
                    design[:, d] = outer.flatten()
                for c in range(p * n):
                    target = tensor[:, :, c].flatten()
                    try:
                        W[c, :], residuals, rank, sv = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Check convergence occasionally
                if it % 20 == 0:
                    # Compute error efficiently using einsum or loops
                    err = 0.0
                    # T_recon[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                    # We only care about the 54 non-zero entries
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = 0.0
                                for d in range(r):
                                    val += U[a, d] * V[b, d] * W[c, d]
                                err += (val - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    
                    if err < 1e-8:
                        break
        
        return best_err, best_U, best_V, best_W
    
    # Try various ranks, but limit the search to ensure we finish in time
    # We know rank 54 works. Let's try a few lower ranks quickly.
    # If we don't find a better one fast, we return schoolbook.
    start_time = time.time()
    time_limit = 100 # Leave 20s for safety
    
    for r in range(schoolbook_rank - 5, max(schoolbook_rank - 10, 0), -1):
        if time.time() - start_time > time_limit:
            break
        err, U, V, W = try_decomposition(r, num_restarts=2, max_iter=50)
        if err < 1e-8:
            # Found a valid decomposition
            # Convert to list of lists with integer/half-integer entries
            # Round to nearest 0.5
            U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
            V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
            W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
            
            # Verify the rounded decomposition
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rounded[i*m+j][d] * V_rounded[j*p+k][d] * W_rounded[k*n+i][d] for d in range(r))
                        if abs(val - 1.0) > 1e-6:
                            verified = False
                            break
                    if not verified:
                        break
                if not verified:
                    break
            
            if verified:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    # Return schoolbook
    rank = schoolbook_rank
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

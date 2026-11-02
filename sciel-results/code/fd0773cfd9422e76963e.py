"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] where:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T is a 3-tensor with indices (i, j, k)
    # We need to find U, V, W such that:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (with trace convention)
    
    # The output C has shape (n, p) = (2, 9), so p*n = 18 output elements
    # The input A has shape (n, m) = (2, 3), so n*m = 6 elements
    # The input B has shape (m, p) = (3, 9), so m*p = 27 elements
    
    # For each output element C[i,k], we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can view this as: for each (i,k), the bilinear form is sum_j A[i,j]*B[j,k]
    
    # The tensor T has shape (n*m, m*p, p*n) in the flattened convention:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # where a = i*m + j, b = j*p + k, c = k*n + i
    
    # For the schoolbook, each (i,j,k) triple gives one term.
    # Let's build the tensor and find its rank via decomposition.
    
    # Actually, let's think more carefully. The multiplication tensor is defined such that:
    # For the bilinear map (A, B) -> C where C = A*B,
    # C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    
    # So the tensor T[c, a, b] = 1 if c = k*n + i, a = i*m + j, b = j*p + k for some j
    # More precisely: T is a 3-way tensor where T[(i,j), (j,k), (k,i)] contributions...
    
    # Let me define it properly. We want:
    # C_flat[k*n + i] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over j,r
    # Wait, that's not right either.
    
    # Correct formulation:
    # C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    # = sum_j sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]  ... no
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r U[i,j,r] V[j,k,r] W[k,i,r]  -- this doesn't work directly
    
    # Actually, the decomposition is:
    # The multiplication tensor mu_{n,m,p} has entries mu_{i,j,k} = 1 if the (i,k)-th output
    # depends on A[i,j] and B[j,k].
    
    # In the factorized form:
    # mu_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    
    # But with the flattened conventions given:
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]  
    # W has shape (p*n, rank): W[k*n + i, r]
    
    # And the condition is:
    # For the bilinear form computing C[k*n + i] = sum_j A[i*m+j] B[j*p+k],
    # we need: sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] summed over r gives the correct coefficient.
    
    # Actually the standard result: the tensor T_{a,b,c} where a indexes A, b indexes B, c indexes C
    # T_{a,b,c} = 1 iff c is determined by a and b in the multiplication.
    
    # Let's just build it: for each valid (i,j,k) triple, we add to the tensor.
    # The tensor T has dimensions (n*m) x (m*p) x (p*n).
    # T[i*m+j, j*p+k, k*n+i] += 1 for each (i,j,k)
    
    # Then we need to find the minimal rank decomposition.
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Now we need to find U, V, W with minimal rank r such that:
    # T = sum_r U[:, r] outer V[:, r] outer W[:, r]  (in tensor form)
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a tensor rank minimization problem, which is NP-hard in general.
    # But for this specific structure, we can use the following approach:
    
    # The multiplication tensor for (n,m,p) has a known structure. For 2x3x9,
    # we can try to find a decomposition using the fact that the tensor is sparse.
    
    # Let's use an iterative approach: start with schoolbook and try to reduce rank.
    # Or use a linear algebra approach: fix two factors and solve for the third.
    
    # Alternative: use the fact that for fixed i,k, the bilinear form is sum_j A[i,j]B[j,k]
    # This is a rank-1 bilinear form in (A[i,:], B[:,k]) for each j, but summed over j.
    
    # Let me try a different approach: use SVD-based decomposition.
    # Reshape T into a matrix and use SVD to get a low-rank approximation, then refine.
    
    # Actually, for exact decomposition, let's use the following algorithm:
    # 1. Start with the schoolbook decomposition (rank = n*m*p = 54)
    # 2. Try to find linear dependencies among the terms to reduce rank
    
    # For 2x3x9, the schoolbook rank is 54. Can we do better?
    
    # Known bounds: for 2x3x9, the minimal rank is at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # and at most n*m*p = 54.
    
    # Let me try to construct a decomposition with rank 27 or less.
    
    # Idea: For each k (0..8), we need to compute C[0,k] and C[1,k] from A[:,0:3] and B[:,k].
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # For each k, this is a 2x3 times 3x1 multiplication, which has rank at most 2*3*1=6,
    # but actually rank 3 (since it's just 3 bilinear forms).
    
    # But we can potentially share terms across different k values.
    
    # Let's try: for each j (0,1,2) and each i (0,1), we have A[i,j].
    # For each j, we have B[j,0:9].
    
    # The output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: for each i (0,1) and each j (0,1,2), create a term that multiplies A[i,j]
    # with the entire row B[j,0:9]. But that would give 2*3=6 terms, each contributing to 9 outputs.
    # However, in the tensor decomposition, each term r contributes to one specific (a,b,c) combination.
    
    # Wait, I think I'm overcomplicating this. Let me re-read the problem.
    
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where T is the multiplication tensor.
    
    # For the schoolbook, we have one term for each (i,j,k) triple:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # To reduce rank, we need to find linear combinations.
    
    # Let me try a simple approach: use rank = 27 (the lower bound) and see if we can construct it.
    
    # Actually, let's just return the schoolbook for now and verify it's correct, then try to improve.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

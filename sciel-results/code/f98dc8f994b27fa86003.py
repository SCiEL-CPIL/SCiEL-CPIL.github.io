import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Kronecker Product Decomposition
    # The task is to multiply A (2x5) by B (5x5).
    # This can be viewed as a block matrix multiplication or leveraging tensor product structure.
    # Specifically, we can decompose the operation into:
    # 1. A "scalar" multiplication part for the 2x1 vs 1x1 case? No, that's not quite right.
    # 2. Let's look at the indices.
    # A is 2x5. B is 5x5. C is 2x5.
    # C[i, k] = sum_j A[i, j] B[j, k]
    
    # We can construct a rank-10 decomposition using the Kronecker product of two simpler decompositions.
    # Consider the multiplication of a (2x1) matrix by a (1x5) matrix? No.
    # Consider the multiplication of a (2x5) matrix by a (5x5) matrix.
    
    # Let's try to construct U, V, W directly using a known low-rank structure or simple search.
    # Rank 10 is a good target.
    # The "Kronecker" hint suggests:
    # If we had (2x1) * (1x1) and (1x5) * (5x5), ranks 2 and 5?
    # Actually, let's just do a fast random search for rank 10 and 11, and fallback to schoolbook.
    # The previous error was in the random search part. We will fix that and optimize.
    
    # Pre-compute target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1
                
    # Reshape T_target to (pn, nm * mp) for easier solving
    # T_target[gamma, alpha, beta]
    T_flat = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    start_time = time.time()
    deadline = start_time + 140 # 140s limit to be safe (150s total)
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # We will try ranks 10, 11, 12, 13, 14, 15. If we find one, we return it.
    # If not, we return schoolbook.
    # To make it fast, we will use a specific construction for rank 10 if possible, or random search.
    
    # Let's try a structured construction for Rank 10 first.
    # The hint mentions Kronecker product of rank-2 (2x1 * 1x1) and rank-5 (1x5 * 5x5).
    # Wait, (2x1) * (1x1) is just vector scaling? Rank 2?
    # (1x5) * (5x5) is row vector times matrix. Rank 5?
    
    # Let's define the primitive decompositions.
    # Prim 1: (2x1) * (1x1) -> (2x1)
    # A1 is 2x1, B1 is 1x1. C1 is 2x1.
    # C1[i, 0] = A1[i, 0] * B1[0, 0]
    # Tensor T1 has shape (2, 1, 2).
    # T1[i, 0, k] = 1 if i==k else 0? No.
    # C1[i] = A1[i] * B1.
    # T1[i, j, k] where i in 0..1, j in 0..0, k in 0..1.
    # C1[i, k] is not defined, it's C1[i]. The output is 2x1.
    # So T1[i, j, k] = 1 if j=0 and i=k? No.
    # The output index is (i, k) where k is the column of the result.
    # For (2x1) * (1x1), the result is 2x1.
    # Result[i, 0] = A[i, 0] * B[0, 0].
    # So T1[i, 0, 0] = 1 for i=0,1? No.
    # T1[i, j, k] where i is row of A, j is col of A/row of B, k is col of B/col of C.
    # T1[i, j, k] = 1 if j==0 and i==k? No, k is always 0.
    # T1[i, 0, 0] = 1 for all i? No.
    # C[i, 0] = A[i, 0] * B[0, 0].
    # So T1[i, 0, 0] = 1.
    # This tensor has rank 1?
    # T1 = sum_r U1[i,r] V1[j,r] W1[k,r].
    # If we set r=0..1?
    # Actually, the rank of (2x1)*(1x1) is 1?
    # C = A B. A is 2x1, B is 1x1.
    # C is 2x1.
    # C[i,0] = A[i,0] B[0,0].
    # This is a rank 1 tensor in the sense of outer product?
    # T[i,j,k] = A[i,j] B[j,k].
    # T[i,0,0] = A[i,0] B[0,0].
    # This can be written as sum_{r=0}^{1} U[i,r] V[0,r] W[0,r]?
    # If we set U[i,0]=A[i,0], V[0,0]=1, W[0,0]=B[0,0]? No, B is variable.
    # The decomposition must be constant.
    # For (2x1)*(1x1), the tensor is T[i,0,0] = 1 for all i? No.
    # The entries of A and B are variables. The tensor T defines the *structure*.
    # T[i,j,k] = 1 if the multiplication involves A[i,j] and B[j,k].
    # For (2x1)*(1x1):
    # i in {0,1}, j in {0}, k in {0}.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C[i,0] = A[i,0] B[0,0].
    # So T[i,0,0] = 1.
    # This tensor is just a vector of ones?
    # Rank of this tensor:
    # T = e_i (vector) * e_j (vector) * e_k (vector)? No.
    # T[i,0,0] = 1.
    # We can write T[i,0,0] = sum_{r=0}^{1} U[i,r] V[0,r] W[0,r].
    # If we pick U[i,0]=1, V[0,0]=1, W[0,0]=1, then T[i,0,0]=1 for all i.
    # So rank is 1?
    # Wait, the "rank-2 decomposition for 2x1 by 1x1" in the prompt might refer to something else or I am misinterpreting.
    # Let's look at (1x5)*(5x5).
    # i in {0}, j in {0..4}, k in {0..4}.
    # C[0,k] = sum_j A[0,j] B[j,k].
    # T[0,j,k] = 1.
    # Rank of this tensor?
    # T[0,j,k] = 1.
    # We can write T[0,j,k] = sum_{r=0}^{4} U[0,r] V[j,r] W[k,r].
    # If we set U[0,r]=1, V[j,r]=delta(j,r), W[k,r]=delta(k,r)?
    # Then sum_r 1 * delta(j,r) * delta(k,r) = 1 if j==k else 0.
    # This gives identity matrix, not all ones.
    # We need T[0,j,k] = 1 for all j,k.
    # This is the all-ones matrix of size 5x5.
    # The tensor rank of the all-ones matrix (viewed as a bilinear form) is 1?
    # T[j,k] = 1 = sum_r u_j v_k w? No, it's a 3-tensor.
    # T[i,j,k] for i=0 is 1.
    # T[0,j,k] = 1.
    # We can write 1 = sum_{r=0}^{0} U[0,r] V[j,r] W[k,r] with U[0,0]=1, V[j,0]=1, W[k,0]=1.
    # So rank is 1?
    
    # If both primitives have rank 1, the Kronecker product has rank 1.
    # Total rank 1?
    # Let's check the full problem (2x5)*(5x5).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # T[i,j,k] = 1.
    # Can we write T[i,j,k] = sum_{r=0}^{R} U[i,r] V[j,r] W[k,r]?
    # If R=1, U[i,0] V[j,0] W[k,0] = 1 for all i,j,k.
    # This implies U[i,0], V[j,0], W[k,0] are all non-zero constants.
    # Let U[i,0]=1, V[j,0]=1, W[k,0]=1.
    # Then T[i,j,k] = 1.
    # So the rank is 1??
    
    # WAIT. The matrix multiplication tensor is NOT the all-ones tensor in the standard basis.
    # The tensor T is defined such that C = sum_r U_r V_r W_r.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is <A, B> -> C.
    # The tensor T is the identity map on the tensor product space? No.
    # T is the "multiplication tensor".
    # T[i, j, k] is 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Yes, T[i,j,k] = 1.
    # But the decomposition is:
    # C[i,k] = sum_{r} U[i,r] V[j,r] W[k,r] ... wait, the summation index is r.
    # The formula is:
    # C[i,k] = sum_{r=1}^R U[i,r] V[j,r] W[k,r] is WRONG.
    # The correct formula for tensor decomposition of matrix multiplication is:
    # C[i,k] = sum_{r=1}^R U[i,r] V[j,r] W[k,r] is not the standard form.
    # Standard form:
    # C = sum_{r=1}^R U_r V_r W_r
    # where U_r, V_r, W_r are matrices? No, vectors?
    # Usually:
    # C[i,k] = sum_{r=1}^R U[i,r] V[j,r] W[k,r] is dimensionally inconsistent if j is summed over.
    # The correct identity is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # We want to write this as:
    # C[i,k] = sum_{r=1}^R U[i,r] (sum_{j} V[j,r] A[i,j]) W[k,r] ? No.
    
    # The standard tensor decomposition for matrix multiplication (n x m) * (m x p) is:
    # C[i,k] = sum_{r=1}^R U[i,r] V[j,r] W[k,r] is incorrect notation.
    # It is:
    # C[i,k] = sum_{r=1}^R U_r[i] V_r[j] W_r[k] is also incorrect.
    
    # Let's look at the interface:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # The reconstruction is:
    # T[alpha, beta, gamma] = sum_{r} U[alpha, r] V[beta, r] W[gamma, r]
    # where alpha = i*m + j, beta = j*p + k, gamma = k*n + i.
    
    # So we need:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # And T[other] = 0.
    
    # This is the "multiplication tensor" T.
    # Can we decompose this with rank 1?
    # T[alpha, beta, gamma] = 1 if alpha, beta, gamma are consistent, else 0.
    # Consistency:
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    # This is a sparse tensor. It is NOT the all-ones tensor.
    # The all-ones tensor would be T[alpha, beta, gamma] = 1 for all alpha, beta, gamma.
    # That would correspond to C[i,k] = sum_{j, l, ...} A[i,j] B[l,k] which is not matrix multiplication.
    
    # So, back to the Kronecker hint.
    # The hint suggests a rank 10 solution.
    # Let's try to construct it.
    # If we can't construct it analytically, we can use the random search but fix the bug.
    # The bug was `if residuals and residuals[0] > 1e-9`.
    # `residuals` is an array. `if residuals` fails if empty or ambiguous.
    # Fix: `if len(residuals) > 0 and residuals[0] > 1e-9`.
    
    # Also, the random search was too slow/inefficient.
    # We will implement a faster random search for small ranks (10, 11, 12).
    # And we will add a specific check for the Kronecker product structure if we can figure it out.
    # Given the time, fixing the bug and optimizing the search is safer.
    
    # Optimization:
    # 1. Use sparse matrices? No, dense is fine for R=10, nm=10, mp=25, pn=10.
    # A is (250, 10).
    # Solving 10 systems of 250x10 is fast.
    # 2. Use `np.linalg.solve` if possible? No, A is not square. `lstsq` is correct.
    # 3. Check integer/half-integer condition faster.
    
    # Let's rewrite the search loop with the bug fix.
    
    for R in [10, 11, 12, 13, 14, 15]:
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 2000 # Reduced attempts per rank to allow trying multiple ranks
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            U = np.random.randint(-1, 2, (nm, R))
            V = np.random.randint(-1, 2, (mp, R))
            
            A_3d = U[:, None, :] * V[None, :, :]
            A = A_3d.reshape(nm * mp, R)
            
            W_list = []
            valid = True
            for gamma in range(pn):
                t_gamma = T_flat[gamma]
                try:
                    W_g, residuals, rank, s = np.linalg.lstsq(A, t_gamma, rcond=None)
                    # Bug Fix: Check residuals properly
                    if len(residuals) > 0 and residuals[0] > 1e-9:
                        valid = False
                        break
                    W_g_scaled = W_g * 2
                    W_g_rounded = np.round(W_g_scaled)
                    if np.max(np.abs(W_g_scaled - W_g_rounded)) > 1e-6:
                        valid = False
                        break
                    W_list.append(W_g_rounded / 2.0)
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if valid:
                W = np.vstack(W_list)
                U_float = U.astype(np.float64)
                V_float = V.astype(np.float64)
                W_float = W.astype(np.float64)
                
                T_calc = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
                
                if np.array_equal(T_calc, T_target):
                    found = True
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
        
        if found:
            # Found a solution for this R, return it immediately as it's minimal in our list
            return {"U": best_U, "V": best_V, "W": best_W}
            
    # If no low rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

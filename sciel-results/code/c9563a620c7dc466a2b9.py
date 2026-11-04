"""Search for rank-minimal exact tensor decomposition of 2x3 matrix mult 3x9 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[k*n+i, j*m, j*p+k] style.
    Actually the tensor is T[i,j,k] where output[i,k] = sum_j A[i,j]*B[j,k].
    We reshape: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor element T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i should be 1
    when the indices are consistent (same j in a and b, same i in a and c, same k in b and c).
    Actually the multiplication tensor T is defined as:
    T[i,j,k] = 1 (it's the standard multiplication tensor)
    We flatten: index1 = i*m+j, index2 = j*p+k, index3 = k*n+i
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    # Reconstruct: T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    n, m, p = T.shape
    nm, mp, pn = T.shape
    # Efficient check: for each a,b, compute the vector sum_r U[a,r]*V[b,r]*W[:,r]
    # and compare to T[a,b,:]
    # This is: (U @ V.T) * W element-wise... actually:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = (U * W.T @ ...) no.
    # Let's do: for each r, outer product U[:,r] * V[:,r] * W[:,r]
    # More efficient: M[a,b] = sum_r U[a,r]*V[b,r] = (U @ V.T)[a,b]
    # Then T[a,b,c] = M[a,b] * W[c,r]... no that's not right either.
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Fix c: T[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
    # This isn't simply M[a,b]*W[c,?] because M sums over r.
    # We need: for each c, T[:,:,c] = sum_r U[:,r]*V[:,r]*W[c,r]
    # = sum_r W[c,r] * (U[:,r] * V[:,r])  [outer product per r, scaled by W[c,r]]
    # Let's just do it with einsum or loops for verification.
    # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    # This might be slow for large dimensions but n*m=6, m*p=27, p*n=18, rank up to 54
    # So 6*27*18*rank operations which is fine.
    try:
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    except Exception:
        return False

def _rank_from_factors(U, V, W):
    return U.shape[1]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    T = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p  # 54
    
    # Strategy 1: Try to find lower rank via structured search
    # The multiplication tensor for (2,3,9) - we know rank <= 54 (schoolbook)
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also rank >= (n+m+p-2) by Strassen-type bounds? Not exactly.
    # For (n,m,p), a simple bound is rank >= (n*m + m*p + p*n) / something... 
    # Actually rank >= max(n*m, m*p, p*n) is not quite right either.
    # The tensor is in (R^{nm})* (R^{mp}) * (R^{pn})
    # A simple lower bound: rank >= (nm + mp + pn) / (min(nm,mp,pn))... no.
    # Let's just search.
    
    # Strategy: Random initialization + iterative refinement (like a multiplicative weight / gradient approach)
    # Or: structured low-rank based on Kronecker products
    
    # Idea: Factor as Kronecker-like products.
    # A is 2x3, B is 3x9. 
    # Think of A as 2x3 = (2x1) * (1x3) + ... or block structure.
    # B is 3x9 = 3x3 * 3x1 blocks or 3x(3*3).
    
    # Let's try a structured decomposition:
    # View A as [A1, A2] where A1 is 2x1, A2 is 2x2 (columns)
    # View B as [B1, B2, B3] where each Bj is 3x3
    # Then AB = A1*B1 + A2*B2 (if we split 3 into 1+2)
    # Or AB = sum over column blocks of A
    
    # Actually, let's try: A is 2x3, split into 3 column vectors a1,a2,a3 (each 2-dim)
    # B is 3x9, split into 3 row blocks b1,b2,b3 (each 1x9)
    # AB = a1*b1 + a2*b2 + a3*b3
    # Each ai is 2x1, bi is 1x9. Product is 2x9.
    # This is a sum of 3 rank-1 (in the bilinear sense) terms? No, each term is outer product of 2-dim and 9-dim vectors, which is rank 1 as a matrix but the bilinear complexity...
    
    # The bilinear complexity of computing a*b where a is 2-dim and b is 9-dim to get a 2x9 matrix:
    # That's just 2*9 = 18 multiplications. So total 3*18 = 54. Same as schoolbook.
    
    # Better: Group columns of A and rows of B.
    # If A has columns grouped and B has rows grouped consistently:
    # A = [A1, A2, A3] (each 2x1), B = [B1; B2; B3] (each 1x9)
    # AB = A1*B1 + A2*B2 + A3*B3
    # Each Ai*Bi: 2 scalars * 9 scalars = 18 mults. Total 54.
    
    # To do better, we need to exploit structure. 
    # For 2x3 * 3x9:
    # Note that 2x3 * 3x9 = 2x9.
    # The output has 18 entries.
    # Lower bound on rank: at least 18 (since output has 18 independent entries in general).
    # Actually the rank of the tensor is at least the size of the output = 18? No, that's not a valid lower bound for tensor rank.
    
    # Let me try a different approach: use the fact that we can do 2x3 * 3x3 with rank 9 (schoolbook for that),
    # and then extend to 3x9 by treating 9 as 3*3.
    
    # 2x3 * 3x9: Think of B as 3x3 x 3 (i.e., B = [Bcol1, Bcol2, Bcol3] where each is 3x3)
    # Then AB = [A*Bcol1, A*Bcol2, A*Bcol3]
    # Each A*Bcol_i is 2x3 * 3x3, which has rank 9.
    # Total: 3*9 = 27 if independent, but they share the A matrix.
    # Actually, if we precompute some intermediate products of A, we can share.
    
    # Key insight: A is 2x3. We have 6 entries of A.
    # B has 27 entries.
    # The product depends bilinearly on all 6+27=33 inputs.
    
    # Let's try: rank 27 decomposition.
    # U shape (6, 27), V shape (27, 27), W shape (18, 27)
    # Idea: For each of the 3 column-blocks of B (each 3x3), we do a 2x3*3x3 multiply.
    # 2x3*3x3 schoolbook is rank 9. So 3*9=27 total.
    # But we can potentially share: the A factors are the same across all 3 blocks.
    
    # Construction for rank 27:
    # For block c (c=0,1,2), B_block_c is rows 0-2 of B, columns 3c to 3c+2.
    # Standard 2x3 * 3x3 multiplication for each block.
    # U for block c: same U as 2x3*3x3 schoolbook (size 6x9), repeated 3 times = 6x27
    # V for block c: V_c is 9x9 where V_c[j*3+k, r] = 1 for the r-th term in the 2x3*3x3 schoolbook for block c
    # W for block c: W_c is 18x9 where W_c[k*2+i, r] = 1 for output entry (i, 3c+k)
    
    # Let's build this:
    U_27 = np.zeros((6, 27))
    V_27 = np.zeros((27, 27))
    W_27 = np.zeros((18, 27))
    
    for c in range(3):  # column block of B
        for i in range(2):  # row of A / output
            for j in range(3):  # column of A / row of B
                for k in range(3):  # column of B block
                    r = c * 9 + i * 3 + j * 1 + k * 0  # Need proper indexing
                    # Actually for 2x3 * 3x3: index r = i*3*j... let me be careful.
                    # Schoolbook for (2,3,3): rank 9, index r = i*3 + j*1 + ... no, r = i*3*j? No.
                    # r = i * 3 + j * 3 + k? That's i*3+j*3+k which ranges 0..14, too many.
                    # For (n,m,p)=(2,3,3): rank = n*m*p = 18? No wait.
                    # Schoolbook for (2,3,3): for each (i,j,k) triple, one term. 2*3*3=18 terms.
                    # But we said rank 9 earlier - that was wrong. Rank for 2x3*3x3 is 18 schoolbook.
                    # Hmm, but Strassen can do 2x2*2x2 in 7. For 2x3*3x3, minimum is not trivial.
                    
    # I messed up. Let me redo.
    # Schoolbook for (n,m,p) has rank n*m*p.
    # For (2,3,9): rank = 2*3*9 = 54. That's the seed.
    # For (2,3,3): rank = 2*3*3 = 18.
    
    # So 3 blocks of 2x3*3x3 each with rank 18 gives 3*18=54. No improvement.
    
    # We need to find actual rank reduction.
    
    # Let me try random search for rank r < 54.
    # For small r, random search in the space of rank-r decompositions is hard.
    # But we can try: initialize with random U, V, W of rank r, then iteratively project.
    
    # Alternative: use the fact that the tensor has symmetry or structure.
    
    # Let me try a structured approach:
    # A is 2x3. Think of it as two row vectors a0 (3-dim), a1 (3-dim).
    # B is 3x9. Think of it as three row vectors b0, b1, b2 (each 9-dim).
    # Output C is 2x9: C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,k] = a0 dot B_col_k (where B_col_k is column k of B, a 3-vector)
    # C[1,k] = a1 dot B_col_k
    
    # So C[0] = a0 * B (row vector 3-dim times 3x9 matrix = 1x9)
    # C[1] = a1 * B
    
    # Computing a0*B where a0 is 1x3 and B is 3x9: this is 3*9=27 mults schoolbook.
    # But we can do better: a0*B = [a0[0]*B[0,:] + a0[1]*B[1,:] + a0[2]*B[2,:]]
    # Each term is scalar*vector(9). 
    # The bilinear complexity of 1x3 * 3x9:
    # Inputs: a0 (3 scalars), B (27 scalars). Output: 9 scalars.
    # Lower bound: 9 (output size). 
    # Can we do it in 9? C[k] = sum_j a0[j]*B[j,k]. This is 3 mults per output, 9 outputs = 27.
    # To do better, we'd need to compute intermediate sums.
    # E.g., let s0 = a0[0]*B[0,k] + a0[1]*B[1,k] + a0[2]*B[2,k] for each k. That's 3*9=27.
    # No obvious way to reduce below 27 for a single row.
    
    # But for 2 rows: C[0,k] and C[1,k] both depend on the same B[j,k].
    # C[0,k] = a0[0]*B[0,k] + a0[1]*B[1,k] + a0[2]*B[2,k]
    # C[1,k] = a1[0]*B[0,k] + a1[1]*B[1,k] + a1[2]*B[2,k]
    
    # For each column k of B, we compute a 2x3 * 3x1 product = 2 outputs.
    # That's 2*3=6 mults per column, 9 columns = 54. Same as schoolbook.
    
    # To reduce: exploit that a0 and a1 are fixed (bilinear, so we can precompute combinations).
    # Let u = a0[0]*a1[1] - a0[1]*a1[0] (determinant of 2x2 submatrix)
    # This is a "Strassen-like" trick.
    
    # For 2x2 * 2x2, Strassen uses 7 mults instead of 8.
    # Key identity: 
    # M1 = (a0+a1)*(b0+b1)
    # M2 = a0*(b1-b0)  
    # M3 = (a0+a1)*b0
    # etc.
    
    # Let me try to apply Strassen-like ideas to 2x3 * 3x9.
    
    # Group: think of A as [A_left, A_right] where A_left is 2x2, A_right is 2x1.
    # B as [B_left, B_right] where B_left is 3x6, B_right is 3x3.
    # AB = A_left*B_left + A_right*B_right
    # A_left*B_left: 2x2 * 2x6... wait, dimensions don't match. A_left is 2x2, B_left is 3x6. Can't multiply.
    
    # Let me regroup: A is 2x3. Split columns into (1, 2): A = [A1, A2] where A1 is 2x1, A2 is 2x2.
    # B is 3x9. Split rows into (1, 2): B = [B1; B2] where B1 is 1x9, B2 is 2x9.
    # AB = A1*B1 + A2*B2
    # A1*B1: 2x1 * 1x9 = 2x9. This is 2*9 = 18 mults (each output is product of 2 scalars... no).
    # A1 is 2x1 (2 scalars), B1 is 1x9 (9 scalars). A1*B1: output[i,k] = A1[i]*B1[k]. That's 2*9=18 mults.
    # A2*B2: 2x2 * 2x9. Schoolbook: 2*2*9 = 36 mults.
    # Total: 18+36 = 54. No gain.
    
    # Use Strassen for 2x2*2x9:
    # 2x2 * 2x9 can be seen as two 2x2*2x4 + one 2x2*2x1, or just apply Strassen to 2x2 part.
    # Strassen for 2x2 * 2x2 uses 7 mults. For 2x2 * 2x9, we can do:
    # Let A2 = [[a,b],[c,d]], B2 = [[e,f,...],[g,h,...]] (2x9)
    # Standard: 4*9 = 36 mults.
    # Strassen-style: 
    # M1 = (a+d)*(e+g) -- but B is 2x9, so e+g is 1x9, a+d is scalar. Product is 1x9.
    # This is getting complicated.
    
    # Let me just try random search with a reasonable rank.
    
    def try_random_rank(r, n_iter=1000, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalization: the decomposition is only defined up to scaling of columns.
        # We want to find U,V,W such that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a factorization of a tensor.
        
        # Iterative refinement: for each r, fix others and solve for one.
        # Fix V,W, solve for U: T[:,:,c] for each c... this is a linear system.
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Fix V,W: for each (b,c), T[:,b,c] = U @ (V[b,:]*W[c,:])... 
        # Let M_r = V[:,r]*W[:,r] (element-wise product of column r of V and column r of W)
        # Then T[:,b,c] = sum_r U[:,r] * V[b,r]*W[c,r]
        # Hmm, this is: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        # Let P[b,c,r] = V[b,r]*W[c,r]. Then T[a,b,c] = sum_r U[a,r]*P[b,c,r]
        # For fixed b,c: T[:,b,c] = U @ P[b,c,:]^T
        # This is a linear system: for each (b,c), U @ p_bc = t_bc where p_bc = P[b,c,:] and t_bc = T[:,b,c]
        # We have many equations (m*p*p*n = 27*18=486 equations) for n*m=6 unknowns per... 
        # Actually U is 6xr, so 6r unknowns. 486 equations. If r >= 1, overdetermined.
        
        # This is a tensor rank factorization, which is generally hard (NP-hard in general).
        # For small tensors, we can try.
        
        # Let me try a different approach: use the known structure.
        # The tensor T has T[a,b,c]=1 only when a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k.
        # Number of nonzero entries: n*m*p = 54.
        # Tensor rank >= number of nonzero entries / (max row/col/... )... not a useful bound.
        
        # Let me try rank = 27 and see if random search can find it.
        # Actually, let me try a specific structured decomposition.
        
        # Insight: 2x3 * 3x9. Note that 3 = 1+2.
        # A = [a_col0, a_col1, a_col2] (each 2-dim column)
        # B = [b_row0; b_row1; b_row2] (each 1x9 row)
        # AB = a_col0*b_row0 + a_col1*b_row1 + a_col2*b_row2
        # Each a_col_i * b_row_i is a 2x9 outer product (rank 1 matrix, but bilinearly it's 2*9=18 mults).
        # Total: 3*18 = 54.
        
        # To reduce: combine a_col0*b_row0 + a_col1*b_row1.
        # This is a 2x9 + 2x9 = 2x9 matrix.
        # (a0*b0 + a1*b1)[i,k] = a0[i]*b0[k] + a1[i]*b1[k]
        # = sum_i' a[i',i]... no.
        # Let a0 = [a00, a01]^T, a1 = [a10, a11]^T
        # b0 = [b00,...,b08], b1 = [b10,...,b18]
        # Output[0,k] = a00*b0[k] + a10*b1[k]
        # Output[1,k] = a01*b0[k] + a11*b1[k]
        
        # Can we compute this in fewer than 36 mults?
        # We have 4*9 = 36 scalar multiplications in schoolbook.
        # Strassen-like: 
        # Let u0 = a00+a10, u1 = a01+a11
        # Let v0 = b0-b1 (9-dim vector), v1 = b0+b1 (9-dim vector)
        # Then:
        # a00*b0 + a10*b1 = ?
        # (a00+a10)*(b0+b1) = a00*b0 + a00*b1 + a10*b0 + a10*b1
        # (a00-a10)*(b0-b1) = a00*b0 - a00*b1 - a10*b0 + a10*b1
        # Adding: 2*a00*b0 + 2*a10*b1. So a00*b0+a10*b1 = [M1+M2]/2 where M1=(a00+a10)*(b0+b1), M2=(a00-a10)*(b0-b1).
        # But M1 and M2 are vector products (scalar * 9-vector), each requiring 9 mults.
        # So 2 vector products = 18 mults, giving a00*b0+a10*b1.
        # Similarly for a01*b0+a11*b1: another 18 mults.
        # Total for this pair: 36 mults. Same as schoolbook (36). No gain.
        
        # Hmm. The Strassen trick for 2x2*2x2 saves 1 mult (8->7).
        # For 2x2*2x9, schoolbook is 36. Can we do better?
        # 2x2*2x9: A=[[a,b],[c,d]], B=[[e1,...,e9],[f1,...,f9]]
        # Output[0,k] = a*e[k]+b*f[k], Output[1,k] = c*e[k]+d*f[k]
        # Schoolbook: 4*9=36 mults.
        # Strassen: 
        # M1 = (a+d)*(e+f) -- e+f is 1x9, so 9 mults
        # M2 = (a-d)*(e-f) -- 9 mults  
        # M3 = a*e -- 9 mults
        # M4 = d*f -- 9 mults
        # Output[0,k] = (M1[k]+M2[k])/2 ... no that gives a*e+d*f.
        # We need a*e+b*f and c*e+d*f.
        # Let me define:
        # M1 = (a+b)*(e+f) : 9 mults (scalar * 9-vector)
        # M2 = a*e : 9 mults
        # M3 = b*f : 9 mults
        # M4 = (c+d)*(e+f) : 9 mults
        # M5 = c*e : 9 mults
        # M6 = d*f : 9 mults
        # Output[0,k] = M2[k]+M3[k] = a*e[k]+b*f[k]
        # Output[1,k] = M5[k]+M6[k] = c*e[k]+d*f[k]
        # Total: 54 mults. Worse!
        
        # The issue is that scalar*vector products are expensive.
        # Strassen works well when both factors are square because the "extra" multiplications are scalar*scalar.
        
        # For our problem, the structure is 2x3 * 3x9. The "inner" dimension is 3.
        # Let me think of it as: for each of the 3 inner indices j, we compute A[:,j] * B[j,:] which is 2x9.
        # Sum over j.
        
        # Total mults: sum over j of (2*9) = 3*18 = 54.
        
        # To reduce, we need to find a way to compute the sum with fewer mults.
        # This is equivalent to finding a low-rank decomposition of the tensor.
        
        # Let me try: rank 45. Or rank 40. Or rank 36.
        # Actually, let me just try to find ANY rank below 54 via search.
        
        # Approach: use the "border rank" or try random initializations and refine.
        # For a tensor T in R^I x R^J x R^K, a rank-r decomposition is T = sum_{i=1}^r u_i v_i w_i.
        # We can try to find it by optimizing over u_i, v_i, w_i.
        
        # Since the tensor is small (6x27x18), let me try.
        
        def random_search(r, n_trials=50, n_steps=200):
            for trial in range(n_trials):
                rng = np.random.RandomState(trial + 1000)
                U = rng.randn(n * m, r) * 0.5
                V = rng.randn(m * p, r) * 0.5
                W = rng.randn(p * n, r) * 0.5
                
                # Normalization: we can scale columns freely.
                # Let's normalize so that sum of squares per column is 1 for stability.
                for col in range(r):
                    norm_U = np.linalg.norm(U[:, col])
                    norm_V = np.linalg.norm(V[:, col])
                    norm_W = np.linalg.norm(W[:, col])
                    total = norm_U * norm_V * norm_W
                    if total > 1e-10:
                        U[:, col] /= norm_U
                        V[:, col] /= norm_V
                        W[:, col] /= norm_W
                        # The product norm is now 1, but we need to adjust... 
                        # Actually this normalization makes the optimization harder. Let's not do this.
                
                # Iterative: alternate least squares
                for step in range(n_steps):
                    if time.time() > deadline - 5:
                        break
                    # Update U: fix V, W.
                    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # For fixed b,c: T[:,b,c] = U @ (V[b,:]*W[c,:])
                    # Let P_bc = V[b,:]*W[c,:] (element-wise, shape r)
                    # T[:,b,c] = U @ P_bc
                    # We have many (b,c) pairs. Stack them:
                    # For all (b,c): T[:,b,c] = U @ P_bc
                    # This is: for each a: T[a,b,c] = sum_r U[a,r]*P_bc[r]
                    # Let's build the linear system:
                    # Unknowns: U (6xr)
                    # Equations: for each (b,c) in 27x18=486 pairs, for each a in 6: 486*6=2916 equations
                    # But many are redundant.
                    # Better: for each a, T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])... 
                    # V[:,r] is 27-dim, W[:,r] is 18-dim. Outer product is 27x18.
                    # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                    # This is: T[a] (27x18) = sum_r U[a,r] * O_r where O_r = outer(V[:,r], W[:,r])
                    # Flatten: vec(T[a]) = sum_r U[a,r] * vec(O_r)
                    # For all a: T_flat (6*27*18) = A_ums @ U_flat
                    # where A_ums has columns vec(O_r) for r=0..r-1, repeated for each a.
                    # Actually: T_flat[a*27*18 + b*18 + c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # Let's build matrix M (27*18 x r) where M[b*18+c, r] = V[b,r]*W[c,r]
                    # Then for each a: T[a].flat = U[a,:] @ M... no.
                    # T[a,b,c] = sum_r U[a,r] * M[b*18+c, r]
                    # So T[a].flat = (M.T @ U[a,:]).flat? 
                    # T[a].flat = M.T @ U[a,:] where U[a,:] is 1xr.
                    # T[a] (27x18) with T[a,b,c] = sum_r U[a,r]*M[b,c,r]
                    # = (U[a,:] @ M) ... no. M is 486xr. U[a,:] is 1xr.
                    # T[a,b,c] = U[a,r] * M[b*18+c, r] summed over r.
                    # Let M_bc be row b*18+c of M. T[a].flat = M.T @ U[a,:]^T... 
                    # M.T is xr486. U[a,:]^T is rx1. Product is 486x1 = T[a].flat.
                    # So T[a].flat = M.T @ U[a,:]
                    # For all a: T_flat (6*486) = [M.T, M.T, ..., M.T] (6 blocks) @ U_flat (6r)
                    # This is a linear system. Solve least squares.
                    
                    # Build M: M[b*18+c, r] = V[b,r]*W[c,r]
                    M = np.einsum('br,cr->bcr', V, W).reshape(m * p, r)
                    # M is (27*r) no, M is (27*18, r) = (486, r)
                    
                    # T_flat: for a in 0..5, T[a].flat is T.reshape(6, -1)[a]
                    T_flat = T.reshape(n * m, -1)  # (6, 486)
                    
                    # For each a: T_flat[a] = M.T @ U[a,:]  (486 = 486)
                    # Stack: A_system (6*486, 6r) 
                    # A_system[a*486 + idx, a'*r + r'] = M.T[idx, r'] if a==a' else 0
                    # This is block diagonal.
                    # We can solve for each a independently:
                    # T_flat[a] (486) = M.T (486xr) @ U[a,:] (r)
                    # U[a,:] = lstsq(M.T, T_flat[a])
                    
                    for a in range(n * m):
                        # Solve U[a,:] from M.T @ U[a,:] = T_flat[a]
                        try:
                            U[a, :], residuals, rank_u, sv = np.linalg.lstsq(M.T, T_flat[a], rcond=None)
                        except:
                            pass
                    
                    # Similarly update V:
                    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # For fixed a,c: T[a,:,c] = sum_r U[a,r]*W[c,r] * V[:,r]
                    # Let N_ac[r] = U[a,r]*W[c,r]. T[a,:,c] = sum_r N_ac[r]*V[:,r]
                    # Flatten: T[a,:,c] (27) = V @ N_ac (r) ... V is 27xr, N_ac is r.
                    # T[a,:,c] = V @ N_ac
                    # For all (a,c): T[:,:,c] for c... 
                    # T[a,b,c] for fixed c: (n*m x m*p) = (6x27)
                    # T[:, :, c] = sum_r (U[:,r]*W[c,r]) * V[:,r]^T... 
                    # T[a,b,c] = sum_r (U[a,r]*W[c,r]) * V[b,r]
                    # Let P_r = U[:,r]*W[c,r] (6-dim). T[:,b,c] = sum_r P_r[a]*V[b,r]
                    # T[:,:,c] (6x27) = sum_r P_r * V[:,r]^T = (P_matrix) @ V where P_matrix is 6xr with P_matrix[a,r]=U[a,r]*W[c,r]
                    # So T[:,:,c] = P_c @ V where P_c[a,r] = U[a,r]*W[c,r]
                    # For each c: solve V from P_c.T @ V.T = T[:,:,c].T
                    # V.T (rx27) from P_c (6xr): T[:,:,c].T (27x6) = V.T @ P_c.T
                    # V.T = lstsq(P_c.T, T[:,:,c].T)
                    
                    for c in range(p * n):
                        P_c = U * W[c, :]  # (6, r), P_c[a,r] = U[a,r]*W[c,r]
                        try:
                            V.T, _, _, _ = np.linalg.lstsq(P_c.T, T[:,:,c].T, rcond=None)
                        except:
                            pass
                        V = V  # V.T was updated in place? No, V.T is a view.
                        # Actually V.T = lstsq(...) rebinds V.T which doesn't work.
                        # Let me redo:
                        V_new = np.zeros_like(V)
                        V_new = np.linalg.lstsq(P_c.T, T[:,:,c].T, rcond=None)[0].T
                        V = V_new
                    
                    # Similarly update W:
                    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # For fixed a,b: T[a,b,:] = sum_r U[a,r]*V[b,r] * W[:,r]
                    # Let Q_r = U[a,r]*V[b,r]. T[a,b,:] = sum_r Q_r*W[:,r]
                    # T[a,b,:] (18) = W @ Q (r)
                    # For all (a,b): T[a,b] = W @ Q_ab where Q_ab[r] = U[a,r]*V[b,r]
                    # T (6x27x18) -> for each (a,b): T[a,b] = W @ Q_ab
                    # T.reshape(6*27, 18) = (W stacked) ... 
                    # Let Q_matrix (6*27, r) where Q_matrix[a*27+b, r] = U[a,r]*V[b,r]
                    # T.reshape(-1, 18) = Q_matrix @ W
                    # W = lstsq(Q_matrix, T.reshape(-1,18))
                    
                    Q_matrix = np.einsum('ar,br->abr', U, V).reshape(n * m * m * p, r)
                    try:
                        W, _, _, _ = np.linalg.lstsq(Q_matrix, T.reshape(-1, p * n), rcond=None)
                    except:
                        pass
                
                # Check if we found a valid decomposition
                if _verify(U, V, W, T, tol=1e-8):
                    return U, V, W
            return None
        
        # Try ranks from 27 up to 53
        for r in [27, 28, 30, 32, 36, 40, 45, 48, 50, 52, 53]:
            if time.time() > deadline - 20:
                break
            result = random_search(r, n_trials=20, n_steps=100)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, T, tol=1e-10):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
            if time.time() > deadline - 10:
                break
        
        # Convert to lists with integer/half-integer entries if possible
        # The problem says "integer/half-integer entries"
        # Let's check if our solution has nice entries.
        # If not, we may need to round or find a rational solution.
        
        # For now, return the best we found.
        # Convert numpy arrays to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Try to round to half-integers
        def to_half_integer(x):
            # Round to nearest 0.5
            return round(x * 2) / 2
        
        # Check if rounding preserves validity
        U_rounded = np.array([[to_half_integer(x) for x in row] for row in best_U])
        V_rounded = np.array([[to_half_integer(x) for x in row] for row in best_V])
        W_rounded = np.array([[to_half_integer(x) for x in row] for row in best_W])
        
        if _verify(U_rounded, V_rounded, W_rounded, T, tol=1e-10):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
        
        # If we didn't improve, return schoolbook
        if best_rank >= n * m * p:
            best_U, best_V, best_W = _schoolbook(n, m, p)
        
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

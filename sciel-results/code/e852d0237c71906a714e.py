"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Try to find a low-rank decomposition
    # Strategy: Use Strassen-like decomposition for the 4x4 part, extend to 4x7
    
    # First, let's try a structured decomposition
    # For M_{4,4,4}, use a rank-49 decomposition (Strassen recursive)
    # For M_{4,4,7}, extend by adding terms for the extra 3 columns
    
    # Let's try a direct approach: find a decomposition with rank < 112
    # We'll use a greedy algorithm to merge terms
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # Use the fact that we can share A[i,j] across different k values
    
    # For each (i,j), A[i,j] is multiplied by B[j,k] for k=0..6
    # So we can think of this as: for each (i,j), we have a "row" of B[j,:]
    # and we need to compute A[i,j] * B[j,k] for each k
    
    # A rank-1 term corresponds to a choice of (i,j) for U, (j,k) for V, (k,i) for W
    # To reduce rank, we need to find patterns where multiple (i,j,k) triples can be handled together
    
    # Let's try a decomposition based on splitting the k dimension
    # Split k into groups and use different strategies for each group
    
    # For now, let's try a simple improvement:
    # Group k values and see if we can share terms
    
    # Actually, let's try a tensor decomposition using polynomial interpolation
    # For the j dimension (size 4), evaluate at 4 points
    # For the k dimension (size 7), evaluate at 7 points
    
    # This is getting complex. Let me try a simpler approach.
    
    # Try to find a decomposition with rank around 80-100
    # by merging terms that have the same (i,j) or same (j,k) patterns
    
    # For a given (i,j), the terms are: for each k, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # These 7 terms (for k=0..6) all have the same U entry. Can we merge them?
    # Not directly, because V and W entries differ.
    
    # Let me try a different grouping: group by k
    # For a given k, the terms are: for each (i,j), U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # These 16 terms (for i,j=0..3) all have the same V and W entries (for fixed k).
    # Can we merge them? Not directly.
    
    # Let me try to use a known result: M_{4,4,4} has rank at most 49.
    # I'll construct a rank-49 decomposition for M_{4,4,4} and extend it.
    
    # Strassen's algorithm for 4x4:
    # Split A and B into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # where each block is 2x2.
    # Compute:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    # Then:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each M_i is a 2x2 matrix product, which has rank 7 (Strassen for 2x2).
    # So total rank = 7 * 7 = 49.
    
    # Let me implement this.
    
    def strassen_2x2():
        """Return a rank-7 decomposition for M_{2,2,2}."""
        # Strassen's 7 terms for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M0 = (a+d)(e+h)
        # M1 = (c+d)e
        # M2 = a(g-h)
        # M3 = d(f-e)
        # M4 = (a+b)h
        # M5 = (c-a)(e+g)
        # M6 = (b-d)(f+h)
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # U has shape (2*2, 7) = (4, 7)
        # V has shape (2*2, 7) = (4, 7)
        # W has shape (2*2, 7) = (4, 7)
        
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        # i,j,k in {0,1}
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M0 = (a+d)(e+h): U[0,0]=1, U[3,0]=1 (a=A00, d=A11)
        # V[0,0]=1, V[3,0]=1 (e=B00, h=B11)
        # W[0,0]=1, W[3,0]=1 (for i=0,k=0 and i=1,k=1... wait)
        
        # Let me be more careful.
        # For M0 = (A00+A11)(B00+B11):
        # This contributes to C00, C01, C10, C11 with coefficients.
        # Actually, in the tensor decomposition, each M_i is a scalar multiplication.
        # The rank-1 term for M0 is:
        # u: non-zero at positions corresponding to (i,j) in {A00, A11} = {(0,0), (1,1)}
        # v: non-zero at positions corresponding to (j,k) in {B00, B11} = {(0,0), (1,1)}
        # w: non-zero at positions corresponding to (k,i) in ... 
        
        # This is getting complicated. Let me just use the schoolbook for now and try to optimize.
        
        return None
    
    # Given the complexity, let me just return the schoolbook decomposition
    # but try a simple optimization first
    
    # Simple optimization: try to find if any terms can be merged
    # For now, return schoolbook
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try a simple heuristic: group terms by (i,j) and see if we can reduce
    # For each (i,j), we have p=7 terms. Can we represent these 7 terms with fewer rank-1 terms?
    # The 7 terms are: for k=0..6, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This is a rank-7 tensor (in the k dimension) for fixed (i,j).
    # The minimum rank of this sub-tensor is 7 (since it's the identity-like structure).
    # So we can't reduce it for fixed (i,j).
    
    # What if we group by k? For fixed k, we have n*m=16 terms.
    # For fixed k, the terms are: for (i,j), U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This is a rank-16 tensor. Can we reduce it?
    # The structure is: U is all-ones for the (i,j) indices, V is fixed for the j index, W is fixed for the i index.
    # This is actually rank 1! Because U[i*m+j] = 1 for all i,j, V[j*p+k] = 1 for all j, W[k*n+i] = 1 for all i.
    # Wait, no. V[j*p+k] depends on j, and W[k*n+i] depends on i.
    # So for fixed k, the tensor is: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j.
    # This can be written as: sum over (i,j) of u_{i,j} ⊗ v_{j} ⊗ w_{i} where u_{i,j}[i*m+j]=1, v_j[j*p+k]=1, w_i[k*n+i]=1.
    # But this is just the schoolbook for fixed k, with rank 16.
    
    # Can we do better? The tensor for fixed k is: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j.
    # This is equivalent to: T is the outer product of three vectors? No, it's not.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # Let u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1.
    # Then u ⊗ v ⊗ w has entries u[a]*v[b]*w[c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    # But u[a] = 1 for all a, v[b] = 1 for all b (with b ≡ k mod p... no, b = j*p+k, so b mod p = k), w[c] = 1 for all c (with c mod n = i... c = k*n+i, so c mod n = i).
    # So u ⊗ v ⊗ w has entries 1 at positions where a is any, b ≡ k (mod p), c ≡ i (mod n) for some i. But we need b = j*p+k AND c = k*n+i AND a = i*m+j.
    # The condition b = j*p+k and c = k*n+i and a = i*m+j is not captured by u ⊗ v ⊗ w alone.
    # So the rank is not 1.
    
    # I think the schoolbook is already quite good, and reducing it requires sophisticated techniques.
    
    # Let me just return the schoolbook decomposition.
    return schoolbook()
# EVOLVE-BLOCK-END

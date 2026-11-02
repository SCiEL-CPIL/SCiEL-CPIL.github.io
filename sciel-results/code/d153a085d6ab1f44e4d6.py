"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-16 decomposition using:
    # - Strassen's rank-7 algorithm for the first 2 columns (2x2x2)
    # - A rank-9 decomposition for the remaining 3 columns (2x2x3)
    
    # The tensor for 2x2x5 multiplication: T[i,j,k] where
    # i in {0,1} (rows of A), j in {0,1} (cols of A / rows of B), k in {0,1,2,3,4} (cols of B)
    # The output C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the trace convention: W[k*n+i, r]
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (10, rank)
    # W has shape (p*n, rank) = (10, rank)
    #
    # The decomposition satisfies: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where T is the 4x10x10 tensor with T[a,b,c] = 1 if the matrix product
    # involves A[a//m][a%m] * B[(a%m)*p + b%p]... 
    #
    # Actually, let me be more careful about the indexing.
    # U is indexed by (i*m + j) where i is row of A, j is col of A
    # V is indexed by (j*p + k) where j is row of B, k is col of B
    # W is indexed by (k*n + i) where k is col of B, i is row of C (same as row of A)
    #
    # The tensor entry T[i*m+j, j*p+k, k*n+i] = 1 (for the product A[i,j]*B[j,k] contributing to C[i,k])
    # All other entries are 0.
    #
    # For a rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # This means: for each (a,b,c) index triple, sum_{r} U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # Strassen for 2x2x2: rank 7
    # The 7 rank-1 terms for multiplying 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    #
    # Strassen's 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5
    #
    # Let me map to our indexing:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices (k*n+i): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    #
    # For Strassen term M1:
    # U part: a00+a11 -> U[0]=1, U[3]=1, others 0
    # V part: b00+b11 -> V[0]=1, V[3]=1, others 0
    # W part: contributes to C00 (+1), C11 (+1) -> W[0]=1, W[3]=1, others 0
    #
    # M2: (a10+a11)*b00
    # U: U[2]=1, U[3]=1
    # V: V[0]=1
    # W: C10 (+1), C11 (+1) -> W[2]=1, W[3]=1
    #
    # M3: a00*(b01-b11)
    # U: U[0]=1
    # V: V[1]=1, V[3]=-1
    # W: C01 (+1), C11 (-1) -> W[1]=1, W[3]=-1
    #
    # M4: a11*(b10-b00)
    # U: U[3]=1
    # V: V[2]=1, V[0]=-1
    # W: C00 (-1), C10 (+1) -> W[0]=-1, W[2]=1
    #
    # M5: (a00+a01)*b11
    # U: U[0]=1, U[1]=1
    # V: V[3]=1
    # W: C01 (+1), C11 (+1) -> W[1]=1, W[3]=1
    #
    # M6: (a10-a00)*(b00+b10)
    # U: U[2]=1, U[0]=-1
    # V: V[0]=1, V[2]=1
    # W: C00 (+1) -> W[0]=1
    #
    # M7: (a01-a11)*(b01+b11)
    # U: U[1]=1, U[3]=-1
    # V: V[1]=1, V[3]=1
    # W: C00 (+1) -> W[0]=1
    #
    # Now for the 2x2x3 part (columns 2,3,4 of B, which is a 2x3 matrix):
    # We need to multiply 2x2 * 2x3.
    # Schoolbook would be rank 6, but we want rank 9 for the combined approach... 
    # Wait, the strategy says rank 9 for the last three columns.
    # Actually, let me reconsider. The strategy says "rank-9 decomposition for the last three columns".
    # But 2x2x3 schoolbook is rank 6. Let me re-read.
    #
    # "a rank-9 decomposition for the last three columns, derived by embedding the 2x2x3 multiplication tensor 
    # into a 2x2x5 framework and applying a specific linear combination of its rank-12 factors to reduce 
    # the contribution to rank 9"
    #
    # This is confusing. Let me think differently.
    #
    # Actually, I think the approach is:
    # - Use Strassen (rank 7) for columns 0,1 of B
    # - For columns 2,3,4 of B, use a decomposition that shares some structure
    #
    # Let me try a direct approach: just use schoolbook for the whole thing but see if we can do better.
    # Schoolbook rank = 2*2*5 = 20.
    #
    # Alternative: partition B's columns into groups and use different algorithms.
    # Columns 0-1: use Strassen (rank 7)
    # Columns 2-4: use schoolbook for 2x2x3 (rank 6)
    # Total: 7 + 6 = 13
    #
    # But wait - can we really just add them? Yes, if the column sets are disjoint.
    # The U vectors for the Strassen part only have non-zero entries in positions corresponding to A's elements.
    # The V vectors for Strassen only have non-zero entries in B columns 0,1.
    # The V vectors for the 2x2x3 part only have non-zero entries in B columns 2,3,4.
    # So the tensor contributions are on disjoint supports in the V dimension, meaning they don't interfere.
    #
    # So rank 13 should work! Let me construct it.
    
    rank = 13
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen block for columns 0,1 (rank 7)
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1: (a00+a11)*(b00+b11) -> C00+1, C11+1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00 -> C10+1, C11+1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: a00*(b01-b11) -> C01+1, C11-1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: a11*(b10-b00) -> C00-1, C10+1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: (a00+a01)*b11 -> C01+1, C11+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6: (a10-a00)*(b00+b10) -> C00+1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # M7: (a01-a11)*(b01+b11) -> C00+1
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for 2x2x3 (columns 2,3,4) - rank 6
    # V indices for these: j*p+k where j in {0,1}, k in {2,3,4}
    # So: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9
    # W indices: k*n+i where k in {2,3,4}, i in {0,1}
    # So: (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7, (4,0)=8, (4,1)=9
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    # Build the full tensor T and check
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute sum of rank-1 tensors
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check if they match
    if not np.allclose(T, T_decomp):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, column] = 1
                    V[j*p+k, column] = 1
                    W[k*n+i, column] = 1
                    column += 1
    else:
        # Convert to lists for the expected output format
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

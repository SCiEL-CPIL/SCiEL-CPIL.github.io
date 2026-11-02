"""Matrix multiplication 2x2x12 via Kronecker/Strassen factorization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Model 2x2x12 as (2x2x2) ⊗ (1x2x6) structure.
    - Use Strassen's rank-7 decomposition for the 2x2x2 part.
    - The 2x12 matrix has 2 rows and 12 cols. We can view it as 2 rows, each of length 12.
    - Alternatively, view the multiplication A(2x2) * B(2x12) = C(2x12).
    
    Let's think about the tensor structure:
    - U has shape (n*m, rank) = (4, rank)
    - V has shape (m*p, rank) = (24, rank)  
    - W has shape (p*n, rank) = (24, rank)
    
    The contraction is: C[k*n+i, j] = sum over a,b of A[i,j] * B[j,k] ... 
    Wait, let me re-read the convention.
    
    "output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k*n + i, r) where k is the row index of B (0..m-1=0..1), 
    i is the column index of A / row index of C (0..n-1=0..1).
    
    Actually, let's be more careful. For matrix multiplication C = A * B where:
    - A is n x m (2x2)
    - B is m x p (2x12)
    - C is n x p (2x12)
    
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k].
    
    The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me think about the standard tensor decomposition for matmul.
    
    For C[i,k] = sum_j A[i,j] B[j,k], the multiplication tensor has entries T[i,j,k] = 1 
    if we're computing C[i,k] from A[i,j] and B[j,k].
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So U[i*m+j, r] extracts A[i,j], V[j*p+k, r] extracts B[j,k], and W[k*n+i, r] 
    places the product into C[i,k].
    
    For schoolbook: r = n*m*p = 2*2*12 = 48. Each term has one non-zero in each.
    
    Now, can we do better?
    
    Key insight: The 2x2x12 multiplication can be seen as 6 independent 2x2x2 multiplications 
    if we group the 12 columns into 6 groups of 2. But that's not quite right because 
    the A matrix is shared across all groups.
    
    Actually, C[i, 2k:s] for k=0..5 gives a 2x2 submatrix. Each such submatrix is 
    A * B[:, 2k:2k+2]. So we have 6 independent 2x2x2 multiplications that share the same A.
    
    If we use Strassen for each 2x2x2, that's 6*7 = 42 terms. But since A is shared, 
    we might be able to share some of the Strassen intermediates.
    
    Strassen's 7 products for 2x2x2:
    Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    
    For our case, we have 6 such blocks. Let B_k be the k-th 2x2 block of B (k=0..5).
    
    For each k, we compute C[:, 2k:2k+2] = A * B_k using Strassen.
    
    The U matrix entries correspond to linear combinations of A's entries.
    The V matrix entries correspond to linear combinations of B_k's entries.
    The W matrix entries correspond to how the products combine into C.
    
    Since A is the same for all 6 blocks, the U part is shared. For each Strassen product, 
    the U coefficient is the same regardless of which block k we're in. So we can have 
    7 unique U patterns, but they appear for each of the 6 blocks, giving 42 terms total.
    
    Wait, but that's just 6 * 7 = 42, which is what was measured. Can we do better?
    
    The knowledge says: "Using Strassen's rank-7 decomposition for 2x2x2 and sharing the 
    A-factor across 6 blocks reduces rank from 48 to 42 Measured: 1 -> 1.143."
    
    So 42 is already achieved. The score is 48/42 = 1.143.
    
    Can we do better than 42? 
    
    The directive mentions a "rank-9 decomposition for 2x2x2" which seems odd since Strassen 
    is rank 7. Maybe it's referring to something else.
    
    Let me think differently. The 2x12 matrix B can be viewed as a 2x2 matrix times a 
    2x6 or something? No, that doesn't make sense dimensionally.
    
    Alternative: Can we use the fact that p=12 is divisible by 6, and use a different 
    factorization?
    
    Actually, let's just implement the 42-rank Strassen-based decomposition since that's 
    known to work and gives score 1.143.
    
    Let me construct it carefully.
    
    n, m, p = 2, 2, 12
    U: (4, rank), V: (24, rank), W: (24, rank)
    
    For each block k (0..5), we have a 2x2x2 Strassen decomposition.
    Block k corresponds to B's columns 2k and 2k+1, and C's columns 2k and 2k+1.
    
    For block k, the 7 Strassen products give 7 terms. Total rank = 6*7 = 42.
    
    U[i*m+j, r] where r indexes the term. For term r corresponding to block k and Strassen 
    product s (s=0..6), U depends only on the Strassen product s (since A is shared).
    
    Let me define the Strassen U coefficients for the 7 products:
    p1: (a+d)(e+h) -> U part: a+d = [1,0,0,1] for A indices (i*m+j: 0,1,2,3)
       A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
       So U for p1: [1, 0, 0, 1]
    p2: (c+d)e -> U: [0, 0, 1, 1]
    p3: a(f-h) -> U: [1, 0, 0, 0]
    p4: d(g-e) -> U: [0, 0, 0, 1]
    p5: (a+b)h -> U: [1, 1, 0, 0]
    p6: (c-a)(e+f) -> U: [-1, 0, 1, 0]
    p7: (b-d)(g+h) -> U: [0, 1, 0, -1]
    
    V coefficients for each product (B indices: B[0,0], B[0,1], B[1,0], B[1,1]):
    For block k, B's entries are at global indices:
    B[0, 2k] = j*p + 2k = 0*12 + 2k = 2k
    B[0, 2k+1] = 2k+1
    B[1, 2k] = 1*12 + 2k = 12 + 2k
    B[1, 2k+1] = 12 + 2k + 1 = 13 + 2k
    
    p1: (e+h) -> V: [1, 0, 0, 1] for local B indices
    p2: e -> V: [1, 0, 0, 0]
    p3: (f-h) -> V: [0, 1, 0, -1]
    p4: (g-e) -> V: [0, 0, 1, -1]
    p5: h -> V: [0, 0, 0, 1]
    p6: (e+f) -> V: [1, 1, 0, 0]
    p7: (g+h) -> V: [0, 0, 1, 1]
    
    W coefficients for each product (C indices: C[0,0], C[0,1], C[1,0], C[1,1]):
    For block k, C's entries are at global indices:
    C[0, 2k] = 2k*n + 0 = 2k*2 + 0 = 4k... wait.
    
    W is indexed by (k*n + i, r). So W[k*n + i, r] where k is the row of B (0 or 1) 
    and i is the column of A / row of C (0 or 1).
    
    Wait, I need to re-read. "W[k*n+i,r]" - so the first index is k*n+i.
    k ranges over... what? In the context of matmul C[i,k] = sum_j A[i,j]B[j,k],
    i is row of C (0..n-1), k is column of C (0..p-1).
    
    So W's first index should be related to (i, k) of C. If it's k*n+i, then for 
    n=2, the index is 2*k + i. k=0..11, i=0..1. So indices 0..23.
    
    C[i, k] where i=0,1 and k=0..11.
    W index = k*n + i = 2k + i.
    
    For block k_block (0..5), C's columns are 2*k_block and 2*k_block+1.
    C[0, 2*k_block] -> W index = 2*(2*k_block) + 0 = 4*k_block
    C[0, 2*k_block+1] -> W index = 2*(2*k_block+1) + 0 = 4*k_block + 2
    C[1, 2*k_block] -> W index = 2*(2*k_block) + 1 = 4*k_block + 1
    C[1, 2*k_block+1] -> W index = 2*(2*k_block+1) + 1 = 4*k_block + 3
    
    W coefficients for each Strassen product:
    p1: C[0,0]+p1, C[1,1]+p1 -> W: [1, 0, 0, 1] for local C
    p2: C[1,0]+p2, C[1,1]-p2 -> W: [0, 0, 1, -1]
    p3: C[0,1]+p3, C[1,1]+p3 -> W: [0, 1, 0, 1]
    p4: C[0,0]+p4, C[1,0]+p4 -> W: [1, 0, 1, 0]
    p5: C[0,0]-p5, C[0,1]+p5 -> W: [-1, 1, 0, 0]
    p6: C[1,1]+p6 -> W: [0, 0, 0, 1]
    p7: C[0,0]+p7 -> W: [1, 0, 0, 0]
    
    Let me verify: 
    C[0,0] = p1 + p4 - p5 + p7 ✓
    C[0,1] = p3 + p5 ✓
    C[1,0] = p2 + p4 ✓
    C[1,1] = p1 - p2 + p3 + p6 ✓
    
    Good. Now let me code this up.
    """
    n, m, p = DIMENSIONS
    rank = 42  # 6 blocks * 7 Strassen products
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen U coefficients (for A: indices 0=a, 1=b, 2=c, 3=d)
    U_coeffs = [
        [1, 0, 0, 1],   # p1: a+d
        [0, 0, 1, 1],   # p2: c+d
        [1, 0, 0, 0],   # p3: a
        [0, 0, 0, 1],   # p4: d
        [1, 1, 0, 0],   # p5: a+b
        [-1, 0, 1, 0],  # p6: c-a
        [0, 1, 0, -1],  # p7: b-d
    ]
    
    # Strassen V coefficients (for B: indices 0=e, 1=f, 2=g, 3=h)
    V_coeffs = [
        [1, 0, 0, 1],   # p1: e+h
        [1, 0, 0, 0],   # p2: e
        [0, 1, 0, -1],  # p3: f-h
        [0, 0, 1, -1],  # p4: g-e
        [0, 0, 0, 1],   # p5: h
        [1, 1, 0, 0],   # p6: e+f
        [0, 0, 1, 1],   # p7: g+h
    ]
    
    # Strassen W coefficients (for C: indices 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1])
    W_coeffs = [
        [1, 0, 0, 1],   # p1: C[0,0]+1, C[1,1]+1
        [0, 0, 1, -1],  # p2: C[1,0]+1, C[1,1]-1
        [0, 1, 0, 1],   # p3: C[0,1]+1, C[1,1]+1
        [1, 0, 1, 0],   # p4: C[0,0]+1, C[1,0]+1
        [-1, 1, 0, 0],  # p5: C[0,0]-1, C[0,1]+1
        [0, 0, 0, 1],   # p6: C[1,1]+1
        [1, 0, 0, 0],   # p7: C[0,0]+1
    ]
    
    for block in range(6):
        for s in range(7):
            r = block * 7 + s
            
            # U: same for all blocks
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = U_coeffs[s][i * m + j]
            
            # V: B indices for this block
            # B[0, 2*block] = 0*p + 2*block = 2*block
            # B[0, 2*block+1] = 2*block + 1
            # B[1, 2*block] = 1*p + 2*block = p + 2*block
            # B[1, 2*block+1] = p + 2*block + 1
            for local_idx in range(4):
                # local_idx 0: B[0, 2*block], 1: B[0, 2*block+1], 2: B[1, 2*block], 3: B[1, 2*block+1]
                if local_idx == 0:
                    global_idx = 2 * block
                elif local_idx == 1:
                    global_idx = 2 * block + 1
                elif local_idx == 2:
                    global_idx = p + 2 * block
                else:
                    global_idx = p + 2 * block + 1
                V[global_idx, r] = V_coeffs[s][local_idx]
            
            # W: C indices for this block
            # C[0, 2*block] -> W index = (2*block)*n + 0 = 4*block
            # C[0, 2*block+1] -> W index = (2*block+1)*n + 0 = 4*block + 2
            # C[1, 2*block] -> W index = (2*block)*n + 1 = 4*block + 1
            # C[1, 2*block+1] -> W index = (2*block+1)*n + 1 = 4*block + 3
            for local_idx in range(4):
                if local_idx == 0:
                    global_idx = 4 * block
                elif local_idx == 1:
                    global_idx = 4 * block + 2
                elif local_idx == 2:
                    global_idx = 4 * block + 1
                else:
                    global_idx = 4 * block + 3
                W[global_idx, r] = W_coeffs[s][local_idx]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

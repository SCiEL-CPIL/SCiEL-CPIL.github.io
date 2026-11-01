"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x14 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, 2): row of output (2x2 matrix result)
    - j in [0, 2): shared dimension (2x2 times 2x14)
    - k in [0, 14): column of output
    
    T[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    
    We seek a decomposition:
    C[i,k] = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    Wait, let me re-read the interface. The factor matrices are:
    - U: shape (n*m, rank) -> U[i*m + j, r]
    - V: shape (m*p, rank) -> V[j*p + k, r]  
    - W: shape (p*n, rank) -> W[k*n + i, r]
    
    So the decomposition is:
    sum_{j} A[i,j] * B[j,k] = sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to work for all A and B, we need:
    sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i} * delta_{j,j} * ... 
    
    Actually, the standard matrix multiplication tensor is:
    T[i, j, k] = A[i,j] * B[j,k]
    
    And we want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    But with the index conventions given:
    - U has indices (i*m + j, r) so U[i*m+j, r]
    - V has indices (j*p + k, r) so V[j*p+k, r]
    - W has indices (k*n + i, r) so W[k*n+i, r]
    
    The tensor equation is:
    A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to hold for ALL A and B, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triple
    and 0 otherwise.
    
    But wait, A[i,j] * B[j,k] is just a product of two independent variables.
    The tensor T has T[i,j,k] = 1 for all valid (i,j,k) because A[i,j]*B[j,k] 
    is always a valid term.
    
    So we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    This is a rank-r decomposition of the all-ones tensor of size n*m*p.
    
    The minimal rank of the all-ones tensor of size n*m*p is... well, for the 
    standard matrix multiplication tensor, the rank is at most n*m*p (schoolbook)
    and can be lower using algorithms like Strassen.
    
    For 2x2 * 2x14, we can use Strassen's algorithm on the 2x2 part and then 
    handle the 14 columns.
    
    Actually, let me think about this differently. The matrix multiplication 
    (2x2) * (2x14) can be decomposed as:
    
    Let A be 2x2, B be 2x14.
    C = A * B is 2x14.
    
    C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    We can use Strassen for the 2x2 case. Strassen uses 7 multiplications:
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1] - B[1,1])
    M4 = A[1,1] * (B[1,0] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    But this is for 2x2 * 2x2. For 2x2 * 2x14, we can apply Strassen to each 
    pair of columns, or we can find a more efficient decomposition.
    
    Actually, the simplest approach: use the schoolbook decomposition which has 
    rank n*m*p = 2*2*14 = 56.
    
    Can we do better? Let's think about the structure.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear form. The tensor T[i,j,k] = 1 for all (i,j,k).
    
    We need to decompose this tensor into a sum of rank-1 tensors.
    
    A rank-1 tensor has the form: U[i,j] * V[j,k] * W[i,k]
    
    Wait, that's not quite right. Let me re-examine.
    
    The decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So each term r contributes:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to equal 1 for all (i,j,k), we need a set of such products that sum to 1.
    
    One simple decomposition: for each (i,j,k), have a separate term r_{ijk} where:
    U[i*m+j, r_{ijk}] = 1, V[j*p+k, r_{ijk}] = 1, W[k*n+i, r_{ijk}] = 1
    and all other entries are 0.
    
    This gives rank = n*m*p = 56.
    
    Can we share terms? Yes, if we can find U, V, W such that the sum of products 
    equals 1.
    
    For example, if we set all U, V, W entries to 1/sqrt(rank), then each product is 
    1/rank, and we need rank terms to sum to 1. But this doesn't work because the 
    products would be the same for all (i,j,k).
    
    Let me think about this more carefully. We want:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    This is equivalent to saying that the tensor T (all ones) has a decomposition 
    of rank r.
    
    The minimal rank of the all-ones tensor of size n*m*p is not trivial. For 
    n=m=p=2, it's 7 (Strassen). For n=2, m=2, p=14, we need to find the minimal rank.
    
    One approach: use the fact that the tensor is separable in some sense.
    
    T[i,j,k] = 1 = (1/2) * (1 + 1) ... no, that doesn't help.
    
    Actually, T[i,j,k] = 1 can be written as:
    T[i,j,k] = (1/2) * [f(i,j) * g(j,k) * h(i,k)]
    
    where f, g, h are chosen appropriately. But this only gives rank 1 if 
    f(i,j) * g(j,k) * h(i,k) = 2 for all (i,j,k), which is not possible with 
    non-zero f, g, h.
    
    Let me try a different approach. Consider:
    T[i,j,k] = (1/4) * (1 + (-1)^i) * (1 + (-1)^j) * (1 + (-1)^k)
    
    No, that's not right either.
    
    Let's use the following identity:
    1 = (1/2) * (1 + 1) = (1/2) * (e^{i0} + e^{i0}) ... this is getting complicated.
    
    A better approach: use the fact that 1 = (1/2) * (1 + 1), and decompose 
    each dimension.
    
    For a 2x2x2 tensor, we can use:
    T[i,j,k] = (1/8) * sum_{a=0}^{1} sum_{b=0}^{1} sum_{c=0}^{1} 
               (1 + (-1)^{i+a}) * (1 + (-1)^{j+b}) * (1 + (-1)^{k+c})
    
    No, that's not right. Let me think again.
    
    Actually, the simplest way to get a lower rank is to use the structure of 
    matrix multiplication. For (2x2) * (2x14), we can think of it as 7 independent 
    (2x2) * (2x2) multiplications (for columns 0-1, 2-3, etc.), but that's not 
    quite right because the 2x14 matrix has 14 columns, not 7 pairs.
    
    Wait, we can group the 14 columns into 7 pairs of 2 columns each. For each 
    pair of columns (2k, 2k+1), we have a (2x2) * (2x2) multiplication, which can 
    be done with rank 7 (Strassen). So total rank = 7 * 7 = 49.
    
    Let me verify this. For columns 2k and 2k+1:
    C[i, 2k] = A[i,0]*B[0,2k] + A[i,1]*B[1,2k]
    C[i, 2k+1] = A[i,0]*B[0,2k+1] + A[i,1]*B[1,2k+1]
    
    This is exactly a (2x2) * (2x2) multiplication where the second matrix has 
    columns [B[:,2k], B[:,2k+1]].
    
    So we can use Strassen for each of the 7 column pairs, giving rank 7*7 = 49.
    
    Let me implement this.
    
    Strassen's algorithm for (2x2) * (2x2):
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now, for each column pair (2k, 2k+1), we have:
    A is the same 2x2 matrix
    B_k = [[B[0,2k], B[0,2k+1]], [B[1,2k], B[1,2k+1]]]
    
    We apply Strassen to (A, B_k) for each k = 0, ..., 6.
    
    This gives 7 * 7 = 49 terms.
    
    Let me construct the U, V, W matrices.
    
    For each column pair k (0 to 6) and each Strassen term s (0 to 6), we have a 
    rank-1 term.
    
    For Strassen term s in column pair k:
    - The U part depends on A[i,j]
    - The V part depends on B[j, 2k] or B[j, 2k+1]
    - The W part depends on the output C[i, 2k] or C[i, 2k+1]
    
    Let me list the Strassen terms:
    s=0: M1 = (a + d) * (e + h) -> U uses (a+d), V uses (e+h), W contributes to C[0,0] and C[1,1]
    s=1: M2 = (c + d) * e -> U uses (c+d), V uses e, W contributes to C[1,0]
    s=2: M3 = a * (f - h) -> U uses a, V uses (f-h), W contributes to C[0,1] and C[1,1]
    s=3: M4 = d * (g - e) -> U uses d, V uses (g-e), W contributes to C[0,0] and C[1,0]
    s=4: M5 = (a + b) * h -> U uses (a+b), V uses h, W contributes to C[0,0] and C[0,1]
    s=5: M6 = (c - a) * (e + f) -> U uses (c-a), V uses (e+f), W contributes to C[1,1]
    s=6: M7 = (b - d) * (g + h) -> U uses (b-d), V uses (g+h), W contributes to C[0,0] and C[0,1]
    
    Wait, I need to be more careful. Let me re-derive.
    
    For a single (2x2) * (2x2) multiplication with A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    
    M1 = (a+d)*(e+h)
    M2 = (c+d)*e
    M3 = a*(f-h)
    M4 = d*(g-e)
    M5 = (a+b)*h
    M6 = (c-a)*(e+f)
    M7 = (b-d)*(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    So for each (i,j,k) in the original problem, where k is the column index:
    - If k is even (k=2m), it corresponds to column 0 of B_m
    - If k is odd (k=2m+1), it corresponds to column 1 of B_m
    
    For the U matrix, the entry U[i*m + j, r] depends on which Strassen term we're using.
    For the V matrix, the entry V[j*p + k, r] depends on which column of B we're using.
    For the W matrix, the entry W[k*n + i, r] depends on which output element we're computing.
    
    Let me define the Strassen terms more explicitly.
    
    For column pair m (0 to 6):
    B_m = [[B[0,2m], B[0,2m+1]], [B[1,2m], B[1,2m+1]]]
    So e = B[0,2m], f = B[0,2m+1], g = B[1,2m], h = B[1,2m+1]
    
    Strassen term s=0: M1 = (a+d)*(e+h)
    - U: a+d means U[0*m+0, r] = 1, U[0*m+1, r] = 0, U[1*m+0, r] = 0, U[1*m+1, r] = 1
      Wait, a = A[0,0], d = A[1,1]. So a+d = A[0,0] + A[1,1].
      U[0*m+0, r] = 1 (for A[0,0])
      U[1*m+1, r] = 1 (for A[1,1])
    - V: e+h = B[0,2m] + B[1,2m+1]
      V[0*p+2m, r] = 1 (for B[0,2m])
      V[1*p+2m+1, r] = 1 (for B[1,2m+1])
    - W: M1 contributes to C[0,0] and C[1,1] of the (2x2) result, which are C[0,2m] and C[1,2m+1]
      W[2m*n+0, r] = 1 (for C[0,2m])
      W[(2m+1)*n+1, r] = 1 (for C[1,2m+1])
    
    This is getting complex. Let me just implement it directly.
    
    rank = 49
    U = [[0] * rank for _ in range(2 * 2)]  # 4 x 49
    V = [[0] * rank for _ in range(2 * 14)]  # 28 x 49
    W = [[0] * rank for _ in range(14 * 2)]  # 28 x 49
    
    For each column pair m (0 to 6) and Strassen term s (0 to 6):
    r = m * 7 + s
    
    Based on the Strassen term, set the appropriate U, V, W entries.
    
    Let me code this up.
    """
    n, m_dim, p = 2, 2, 14
    rank = 49
    U = [[0] * rank for _ in range(n * m_dim)]
    V = [[0] * rank for _ in range(m_dim * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for (2x2) * (2x2)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    for col_pair in range(7):
        k0 = 2 * col_pair
        k1 = 2 * col_pair + 1
        
        # For each Strassen term s (0 to 6):
        for s in range(7):
            r = col_pair * 7 + s
            
            if s == 0:  # M1 = (a+d)*(e+h)
                # U: a + d
                U[0][r] = 1  # A[0,0]
                U[3][r] = 1  # A[1,1]
                # V: e + h = B[0,k0] + B[1,k1]
                V[0 * p + k0][r] = 1  # B[0,k0]
                V[1 * p + k1][r] = 1  # B[1,k1]
                # W: contributes to C[0,k0] and C[1,k1]
                W[k0 * n + 0][r] = 1
                W[k1 * n + 1][r] = 1
            elif s == 1:  # M2 = (c+d)*e
                # U: c + d
                U[2][r] = 1  # A[1,0]
                U[3][r] = 1  # A[1,1]
                # V: e = B[0,k0]
                V[0 * p + k0][r] = 1
                # W: contributes to C[1,k0]
                W[k0 * n + 1][r] = 1
            elif s == 2:  # M3 = a*(f-h)
                # U: a
                U[0][r] = 1  # A[0,0]
                # V: f - h = B[0,k1] - B[1,k1]
                V[0 * p + k1][r] = 1
                V[1 * p + k1][r] = -1
                # W: contributes to C[0,k1] and C[1,k1]
                W[k1 * n + 0][r] = 1
                W[k1 * n + 1][r] = 1
            elif s == 3:  # M4 = d*(g-e)
                # U: d
                U[3][r] = 1  # A[1,1]
                # V: g - e = B[1,k0] - B[0,k0]
                V[1 * p + k0][r] = 1
                V[0 * p + k0][r] = -1
                # W: contributes to C[0,k0] and C[1,k0]
                W[k0 * n + 0][r] = 1
                W[k0 * n + 1][r] = 1
            elif s == 4:  # M5 = (a+b)*h
                # U: a + b
                U[0][r] = 1  # A[0,0]
                U[1][r] = 1  # A[0,1]
                # V: h = B[1,k1]
                V[1 * p + k1][r] = 1
                # W: contributes to C[0,k0] and C[0,k1]
                W[k0 * n + 0][r] = -1
                W[k1 * n + 0][r] = 1
            elif s == 5:  # M6 = (c-a)*(e+f)
                # U: c - a
                U[2][r] = 1  # A[1,0]
                U[0][r] = -1  # A[0,0]
                # V: e + f = B[0,k0] + B[0,k1]
                V[0 * p + k0][r] = 1
                V[0 * p + k1][r] = 1
                # W: contributes to C[1,k1]
                W[k1 * n + 1][r] = 1
            elif s == 6:  # M7 = (b-d)*(g+h)
                # U: b - d
                U[1][r] = 1  # A[0,1]
                U[3][r] = -1  # A[1,1]
                # V: g + h = B[1,k0] + B[1,k1]
                V[1 * p + k0][r] = 1
                V[1 * p + k1][r] = 1
                # W: contributes to C[0,k0]
                W[k0 * n + 0][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

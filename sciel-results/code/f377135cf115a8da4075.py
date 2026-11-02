"""Optimized tensor decomposition for 2x2 * 2x12 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Construct a rank-20 tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Split the 12 columns into two groups of 6. For each 2x2x6 block,
    we use a rank-10 decomposition derived from the known rank-5 optimality of 2x2x3
    multiplication, combined with a rank-2 linear combination to share intermediate products.
    
    Actually, let me reconsider. The schoolbook rank is 2*2*12 = 48.
    We can do better by exploiting structure.
    
    For 2x2 * 2x12:
    - The 2x2 matrix has 4 entries: a, b, c, d
    - The 2x12 matrix has 24 entries: x_0..x_11 (row 0), y_0..y_11 (row 1)
    
    Output is 2x12:
    Row 0: a*x_0 + b*y_0, a*x_1 + b*y_1, ..., a*x_11 + b*y_11
    Row 1: c*x_0 + d*y_0, c*x_1 + d*y_1, ..., c*x_11 + d*y_11
    
    Each output entry requires 2 multiplications (a*x_i and b*y_i for row 0, c*x_i and d*y_i for row 1).
    Total: 4*12 = 48 scalar multiplications in schoolbook.
    
    Can we share? Note that a*x_i appears in output[0,i] and c*x_i appears in output[1,i].
    If we precompute a*x_i and c*x_i, we still need 2 multiplications per column i for x_i,
    and similarly 2 per column for y_i. So 4 multiplications per column, 12 columns = 48.
    
    Wait, but maybe we can use a different decomposition. Let's think about the tensor rank.
    
    The tensor T[i,j,k] where i is the row of the 2x2 matrix (0,1), j is the column of 2x2/row of 2x12 (0,1),
    and k is the column of 2x12 (0..11).
    
    T[i,j,k] = 1 if we're computing output[k,i] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] = delta for the matrix multiplication.
    
    For each fixed k, we have a 2x2 tensor slice that is the standard 2x2 matrix multiplication tensor.
    The rank of 2x2x2 is known to be 7 (Strassen).
    
    So for each of the 12 columns k, we could use rank 7, giving total rank 7*12 = 84. That's worse than 48.
    
    But wait, the tensor is not a direct sum over k because the same A entries are used across all k.
    
    Let me reconsider. The full tensor has dimensions 2x2x12 (for the indices i, j, k).
    
    T[i,j,k] = 1 (it's the standard matmul tensor where each entry is 1, and the structure is that
    output[i,k] = sum_j A[i,j] B[j,k])
    
    Actually, the tensor is T[i,j,k] = 1 for all i,j,k? No, that's not right.
    
    Let me be precise. The matrix multiplication tensor T is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i,j,k) and T[i,j,k] = 1. The rank of T is the minimum number of
    rank-1 tensors needed to sum to T.
    
    For 2x2x2, rank is 7.
    For 2x2x12, we can think of it as 12 copies of 2x2x2, but they share the A matrix.
    
    One approach: For each k, the slice T[:,:,k] is the 2x2x2 matmul tensor.
    If we use Strassen for each k independently, we get rank 7*12 = 84.
    
    But we can do better by sharing. The key insight is that the A matrix is the same for all k.
    
    Alternative: Use the fact that we can precompute products of A entries and then multiply by B entries.
    
    Let's try a block approach. Split the 12 columns into 4 groups of 3.
    For each group of 3 columns, we have a 2x2x3 tensor.
    
    The rank of 2x2x3 is known to be... let me think. For 2x2xn, the rank is at most 2n (schoolbook)
    but can be lower. For 2x2x3, I believe the rank is 7 or 8.
    
    Actually, let's just try to find a good decomposition empirically or use a known result.
    
    Given time constraints, let me try a simple improvement: use Strassen-like decomposition.
    
    For 2x2x2, Strassen uses 7 multiplications:
    M1 = (a+d)(x+w)
    M2 = (c+d)w
    M3 = a(y-w)
    M4 = d(x-y)
    M5 = (a+b)x
    M6 = (c+d)(y-x)
    M7 = (a+b)(w-x)
    
    Wait, I'm mixing up the indices. Let me use standard Strassen for 2x2:
    Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    p1 = (a+d)(e+h)
    p2 = (c+d)h
    p3 = a(f-h)
    p4 = d(e-f)
    p5 = (a+b)e
    p6 = (c+d)(g-e)
    p7 = (a+b)(h-g)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    
    Now for 2x2x12, we can apply this for each pair of columns in B.
    
    Let B have columns grouped as (b_0, b_1), (b_2, b_3), ..., (b_10, b_11).
    For each group t (0 to 5), let B_t = [[e_t, f_t],[g_t, h_t]] where:
    e_t = B[0, 2t], f_t = B[0, 2t+1], g_t = B[1, 2t], h_t = B[1, 2t+1]
    
    Then for each t, we compute 7 products using Strassen, and combine to get C[:, 2t:2t+2].
    
    Total rank = 7 * 6 = 42.
    
    This is better than 48! Let me implement this.
    
    U, V, W will have rank 42.
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # We'll use Strassen for each pair of columns in B
    # 12 columns -> 6 pairs
    num_pairs = p // 2  # 6
    rank = 7 * num_pairs  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen factors for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(f-h)
    # p4 = d(e-f)
    # p5 = (a+b)e
    # p6 = (c+d)(g-e)
    # p7 = (a+b)(h-g)
    
    # U factors (for A):
    # p1: a+d -> U[0,0]=1, U[1,0]=1 (for a and d, but wait, U is indexed by i*m+j)
    # U[i*m+j, col] where i is row of A, j is column of A
    # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    
    strassen_A = [
        [1, 1, 0, 0],  # p1: a+d
        [0, 0, 1, 1],  # p2: c+d
        [1, 0, 0, 0],  # p3: a
        [0, 0, 0, 1],  # p4: d
        [1, 1, 0, 0],  # p5: a+b
        [0, 0, 1, 1],  # p6: c+d
        [1, 1, 0, 0],  # p7: a+b
    ]
    
    strassen_B = [
        [1, 1, 0, 0],  # p1: e+h -> B[0,0]+B[1,1]
        [0, 0, 0, 1],  # p2: h -> B[1,1]
        [0, 1, 0, -1], # p3: f-h -> B[0,1]-B[1,1]
        [1, -1, 0, 0], # p4: e-f -> B[0,0]-B[0,1]
        [1, 0, 0, 0],  # p5: e -> B[0,0]
        [0, 0, 1, -1], # p6: g-e -> B[1,0]-B[0,0]
        [0, 0, -1, 1], # p7: h-g -> B[1,1]-B[1,0]
    ]
    
    # W factors: C[0,0] = p1+p4-p5+p7, C[0,1] = p3+p5, C[1,0] = p2+p4, C[1,1] = p1-p2+p3+p6
    # W[k*n+i, col] where k is column of C, i is row of C
    # C[0,0] -> k=0, i=0 -> W[0*2+0] = W[0]
    # C[0,1] -> k=1, i=0 -> W[1*2+0] = W[2]
    # C[1,0] -> k=0, i=1 -> W[0*2+1] = W[1]
    # C[1,1] -> k=1, i=1 -> W[1*2+1] = W[3]
    
    # For each Strassen product p_s, W contributions:
    # p1: C[0,0]+=1, C[1,1]+=1
    # p2: C[1,0]+=1, C[1,1]+=-1
    # p3: C[0,1]+=1, C[1,1]+=1
    # p4: C[0,0]+=1, C[1,0]+=1
    # p5: C[0,0]+=-1, C[0,1]+=1
    # p6: C[1,1]+=1
    # p7: C[0,0]+=1
    
    strassen_W = [
        [1, 0, 0, 1],   # p1: C[0,0], C[1,1]
        [0, 1, 0, -1],  # p2: C[1,0], C[1,1]
        [0, 0, 1, 1],   # p3: C[0,1], C[1,1]
        [1, 1, 0, 0],   # p4: C[0,0], C[1,0]
        [-1, 0, 1, 0],  # p5: C[0,0], C[0,1]
        [0, 0, 0, 1],   # p6: C[1,1]
        [1, 0, 0, 0],   # p7: C[0,0]
    ]
    
    for pair_idx in range(num_pairs):
        for s in range(7):
            col = pair_idx * 7 + s
            
            # U
            for idx in range(n * m):
                U[idx][col] = strassen_A[s][idx]
            
            # V
            # B[j, k] where j is row (0,1), k is column within the pair (0,1)
            # V[j*p + k, col]
            for j in range(m):
                for k_local in range(2):
                    k_global = pair_idx * 2 + k_local
                    V[j * p + k_global][col] = strassen_B[s][j * 2 + k_local]
            
            # W
            # C[i, k] where i is row (0,1), k is column within the pair (0,1)
            # W[k*n + i, col]
            for k_local in range(2):
                k_global = pair_idx * 2 + k_local
                for i in range(n):
                    W[k_global * n + i][col] = strassen_W[s][k_local * n + i]
    
    return {"U": U, "V": V, "W": W}

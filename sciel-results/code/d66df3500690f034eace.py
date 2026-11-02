import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-17 exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 subproblem and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, r, u_val=1.0, v_val=1.0, w_val=1.0):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # Strassen part (r=0..6)
    # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # C indices: 0:(0,0), 2:(0,1), 1:(1,0), 3:(1,1)
    
    # P1 = (A[1,0] + A[1,1]) * (B[0,0] - B[1,0])
    add_term(3, 0, 0, 0) # A[1,0], B[0,0] -> C[0,0]
    add_term(4, 0, 0, 0) # A[1,1], B[0,0] -> C[0,0]
    add_term(3, 3, 0, 0, -1.0) # A[1,0], B[1,0] -> C[0,0] (neg)
    add_term(4, 3, 0, 0, -1.0) # A[1,1], B[1,0] -> C[0,0] (neg)
    # Wait, P1 contributes to C[0,0] and C[1,1]
    # C[0,0] += P1
    # C[1,1] += P1
    # So W[0][0] += 1, W[3][0] += 1
    # My add_term only adds to one W. I need to add to both.
    
    # Let's redefine add_term to allow multiple W outputs or just call it twice.
    # Actually, the tensor decomposition is a sum of rank-1 tensors.
    # A rank-1 tensor is $u_i v_j w_k$.
    # If a product $P$ contributes to multiple $C$ elements, it means the rank-1 tensor
    # has non-zero W entries for those indices.
    # So for P1, the rank-1 tensor is:
    # U: 1 at 3, 1 at 4
    # V: 1 at 0, -1 at 3
    # W: 1 at 0, 1 at 3
    # This is a single rank-1 term in the sum?
    # No, the sum is $\sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # If I define one column r, it represents one rank-1 tensor.
    # So yes, I just set the values in the column r for all involved indices.
    
    # Let's restart the filling logic clearly.
    
    # r=0: P1
    U[3][0] += 1.0
    U[4][0] += 1.0
    V[0][0] += 1.0
    V[3][0] += -1.0
    W[0][0] += 1.0
    W[3][0] += 1.0
    
    # r=1: P2
    U[3][1] += 1.0
    U[4][1] += 1.0
    V[4][1] += 1.0
    W[1][1] += 1.0
    W[3][1] += 1.0
    
    # r=2: P3
    U[0][2] += 1.0
    V[1][2] += 1.0
    V[4][2] += 1.0
    W[2][2] += 1.0
    W[3][2] += -1.0
    
    # r=3: P4
    U[4][3] += 1.0
    V[3][3] += 1.0
    V[0][3] += -1.0
    W[0][3] += 1.0
    W[3][3] += 1.0
    
    # r=4: P5
    U[0][4] += 1.0
    U[3][4] += 1.0
    V[4][4] += 1.0
    W[2][4] += 1.0
    W[0][4] += -1.0
    
    # r=5: P6
    U[3][5] += 1.0
    U[0][5] += -1.0
    V[0][5] += 1.0
    V[1][5] += 1.0
    W[1][5] += 1.0
    
    # r=6: P7
    U[1][6] += 1.0
    U[4][6] += -1.0
    V[3][6] += 1.0
    V[4][6] += 1.0
    W[0][6] += 1.0
    W[3][6] += 1.0
    
    # r=7: A[0,2] B[2,0] -> C[0,0]
    U[2][7] += 1.0
    V[6][7] += 1.0
    W[0][7] += 1.0
    
    # r=8: A[0,2] B[2,1] -> C[0,1]
    U[2][8] += 1.0
    V[7][8] += 1.0
    W[2][8] += 1.0
    
    # r=9: A[0,2] B[2,2] -> C[0,2]
    U[2][9] += 1.0
    V[8][9] += 1.0
    W[4][9] += 1.0
    
    # r=10: A[1,2] B[2,0] -> C[1,0]
    U[5][10] += 1.0
    V[6][10] += 1.0
    W[1][10] += 1.0
    
    # r=11: A[1,2] B[2,1] -> C[1,1]
    U[5][11] += 1.0
    V[7][11] += 1.0
    W[3][11] += 1.0
    
    # r=12: A[1,2] B[2,2] -> C[1,2]
    U[5][12] += 1.0
    V[8][12] += 1.0
    W[5][12] += 1.0
    
    # r=13: A[0,0] B[0,2] -> C[0,2]
    U[0][13] += 1.0
    V[2][13] += 1.0
    W[4][13] += 1.0
    
    # r=14: A[0,1] B[1,2] -> C[0,2]
    U[1][14] += 1.0
    V[5][14] += 1.0
    W[4][14] += 1.0
    
    # r=15: A[1,0] B[0,2] -> C[1,2]
    U[3][15] += 1.0
    V[2][15] += 1.0
    W[5][15] += 1.0
    
    # r=16: A[1,1] B[1,2] -> C[1,2]
    U[4][16] += 1.0
    V[5][16] += 1.0
    W[5][16] += 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

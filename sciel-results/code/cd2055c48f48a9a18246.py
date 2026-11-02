import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    start = time.time()
    deadline = start + 130  # safety margin within 150s limit

    n, m, p = 2, 3, 7

    # Schoolbook rank-42 decomposition as guaranteed fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try to find a lower-rank decomposition using block structure
    # Split B into B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1)
    # C = A*B1 + A*B2 + A*B3 + A*B4
    # Each A*Bi where Bi is 3x2: use rank-11 decomposition
    # A*B4 where B4 is 3x1: use rank-6 decomposition
    # Total: 11*3 + 6 = 39

    # For 2x3 * 3x2 multiplication (n=2, m=3, p=2):
    # A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01],[b10,b11],[b20,b21]]
    # C = [[c00,c01],[c10,c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21

    # We can use a Strassen-like decomposition for the 2x2 blocks after splitting
    # Split A into A1 (2x2) = [[a00,a01],[a10,a11]] and A2 (2x1) = [a02,a12]^T
    # Split B into B1 (2x2) = [[b00,b01],[b10,b11]] and B2 (1x2) = [b20,b21]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2*2x2, rank 7 (Strassen)
    # A2*B2: 2x1*1x2, rank 1 (outer product)
    # Total: 7 + 1 = 8? But we need to verify the exact decomposition.

    # Let's use the GQ algorithm approach:
    # The tensor T for matrix multiplication has entries T[i,j,k] = 1 if the output
    # element (i,k) depends on A[i,j] * B[j,k].
    # For n=2, m=3, p=7: T has shape (2,3,7) with T[i,j,k] = 1 for all i,j,k.
    # Actually, the matrix multiplication tensor M_{n,m,p} has:
    # M[i,j,k,l] = delta_{j,l} * delta_{i,k} ... no.
    # The correct tensor: for each output C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's a 3-tensor of size n x m x p)
    # where T[i,j,k] indicates the coefficient of A[i,j]*B[j,k] in C[i,k].
    # Actually T[i,j,k] = 1 for all valid i,j,k since each product A[i,j]*B[j,k]
    # contributes to exactly one output C[i,k].

    # The matricization: flatten along different modes.
    # Mode-1 flattening: shape (n, m*p) -> U-like
    # Mode-2 flattening: shape (m, n*p) -> V-like  
    # Mode-3 flattening: shape (p, n*m) -> W-like

    # For the standard matrix multiplication tensor, all matricizations have
    # full rank = min of the two dimensions.
    # Mode-1: rank = min(2, 21) = 2
    # Mode-2: rank = min(3, 14) = 3
    # Mode-3: rank = min(7, 6) = 6
    # So the tensor rank is at least max(2,3,6) = 6.
    # But we know the actual rank is much higher.

    # The GQ algorithm (Gaussian elimination / canonical polyadic decomposition)
    # works by iteratively finding rank-1 components.

    # Let's implement a direct search for a low-rank decomposition.
    # We'll use the fact that the tensor is structured.

    # Actually, let's try a constructive approach using block matrices.
    # For 2x3 * 3x7:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02, b03, b04, b05, b06],
    #      [b10, b11, b12, b13, b14, b15, b16],
    #      [b20, b21, b22, b23, b24, b25, b26]]
    # C[i,k] = sum_j A[i,j] * B[j,k]

    # We can group columns of B. Let's try to use a Kronecker product structure.
    # 7 = 3 + 3 + 1, so split B into B1 (3x3), B2 (3x3), B3 (3x1).
    # C = A*B1 + A*B2 + A*B3

    # For A*B1 where B1 is 3x3: 2x3 * 3x3
    # For A*B2 where B2 is 3x3: 2x3 * 3x3
    # For A*B3 where B3 is 3x1: 2x3 * 3x1

    # 2x3 * 3x3: Can we do better than 18?
    # 2x3 * 3x1: Can we do better than 6?

    # Let's try to find explicit decompositions.

    # For 2x3 * 3x1 (outer product style):
    # C0 = a00*b00 + a01*b10 + a02*b20
    # C1 = a10*b00 + a11*b10 + a12*b20
    # This is just two dot products of length 3.
    # We can use rank 3:
    # r=0: U=[1,0,0], V=[1,0,0], W=[1,0] -> a00*b00
    # r=1: U=[0,1,0], V=[0,1,0], W=[0,1] -> a01*b10  
    # Wait, W should have shape p*n = 1*2 = 2.
    # U shape: n*m = 6, V shape: m*p = 3, W shape: p*n = 2
    # r=0: U[0]=1 (a00), V[0]=1 (b00), W[0]=1 (C0)
    # r=1: U[1]=1 (a01), V[1]=1 (b10), W[1]=1 (C1)
    # r=2: U[2]=1 (a02), V[2]=1 (b20), W[2]=1 (C2) - but C only has 2 elements!
    
    # I'm confusing the indices. Let me be more careful.
    # For the full 2x3*3x7 problem:
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (14, rank)
    # W[k*n+i, r] is the coefficient for C[i,k] in the r-th rank-1 term.

    # Schoolbook: for each (i,j,k), one rank-1 term:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1

    # Let's try to find a better decomposition by exploiting structure.
    # One approach: use the fact that 7 = 4 + 3, and use different decompositions.

    # For now, let's try a simple improvement: use Strassen for 2x2 blocks.
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_left (3x4) and B_right (3x3)
    # C = A_left * B_left + A_right * B_right

    # A_left * B_left: 2x2 * 3x4 -> not standard
    # This is getting complicated. Let's just return the schoolbook for now.
    # The search for a lower-rank decomposition requires more work.

    if time.time() > deadline:
        return best_result

    # Try a simple rank-36 decomposition by splitting B into 3x2, 3x2, 3x2, 3x1
    # and using rank-9 for each 3x2 block and rank-6 for the 3x1 block.
    # 9*3 + 6 = 33? Let's see if we can construct this.

    # Actually, let's try a direct approach: use the Kronecker product structure.
    # The tensor for 2x3*3x7 can be seen as a sum of tensors for smaller multiplications.

    # For now, return the schoolbook decomposition which is guaranteed correct.
    return best_result

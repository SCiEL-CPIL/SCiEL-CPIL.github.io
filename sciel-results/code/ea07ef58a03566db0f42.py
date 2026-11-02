"""Matrix multiplication 4x4x9: Winograd FFT block decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7."""
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # C[0,0]=a00*b00+a01*b10, C[0,1]=a00*b01+a01*b11
    # C[1,0]=a10*b00+a11*b10, C[1,1]=a10*b01+a11*b11
    # U indexes A: [a00, a01, a10, a11]
    # V indexes B: [b00, b01, b10, b11]
    # W indexes C: [c00, c01, c10, c11] (W[k*n+i] where n=2, so c00->W[0], c01->W[1], c10->W[2], c11->W[3])
    # Wait, W[k*n+i]: k is the B row index (0 or 1), i is the A row index (0 or 1)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # W[k*n+i] for k in [0,1], i in [0,1]: W[0]=k0i0, W[1]=k0i1, W[2]=k1i0, W[3]=k1i1
    # But C has indices (i,j) where i is row, j is col. The W index is k*n+i.
    # So W[0] corresponds to (k=0,i=0), W[1] to (k=0,i=1), W[2] to (k=1,i=0), W[3] to (k=1,i=1)
    # The decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta
    # For 2x2x2: n=2,m=2,p=2
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b11
    # p3 = a00*(b01-b11)
    # p4 = a11*(b00-b10)
    # p5 = (a00+a01)*b00
    # p6 = (a10+a11)*b10
    # p7 = (a00-a10)*(b01+b10)
    
    # U[i*2+j, r]:
    # r=0: a00+a11 -> U[0,0]=1, U[3,0]=1
    # r=1: a10+a11 -> U[2,1]=1, U[3,1]=1
    # r=2: a00 -> U[0,2]=1
    # r=3: a11 -> U[3,3]=1
    # r=4: a00+a01 -> U[0,4]=1, U[1,4]=1
    # r=5: a10+a11 -> U[2,5]=1, U[3,5]=1
    # r=6: a00-a10 -> U[0,6]=1, U[2,6]=-1
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[3][5] = 1
    U[0][6] = 1; U[2][6] = -1
    
    # V[j*2+k, r]:
    # r=0: b00+b11 -> V[0,0]=1, V[3,0]=1
    # r=1: b11 -> V[3,1]=1
    # r=2: b01-b11 -> V[1,2]=1, V[3,2]=-1
    # r=3: b00-b10 -> V[0,3]=1, V[2,3]=-1
    # r=4: b00 -> V[0,4]=1
    # r=5: b10 -> V[2,5]=1
    # r=6: b01+b10 -> V[1,6]=1, V[2,6]=1
    
    V[0][0] = 1; V[3][0] = 1
    V[3][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[0][3] = 1; V[2][3] = -1
    V[0][4] = 1
    V[2][5] = 1
    V[1][6] = 1; V[2][6] = 1
    
    # W[k*2+i, r]:
    # C[0,0] = p1+p2-p3-p4 -> W[0,0]=1, W[0,1]=1, W[0,2]=-1, W[0,3]=-1
    # C[0,1] = p1-p2+p3+p4 -> W[1,0]=1, W[1,1]=-1, W[1,2]=1, W[1,3]=1
    # C[1,0] = -p1+p2+p5+p6 -> W[2,0]=-1, W[2,1]=1, W[2,4]=1, W[2,5]=1
    # C[1,1] = p1+p5+p6-p7 -> W[3,0]=1, W[3,4]=1, W[3,5]=1, W[3,6]=-1
    
    W[0][0] = 1; W[0][1] = 1; W[0][2] = -1; W[0][3] = -1
    W[1][0] = 1; W[1][1] = -1; W[1][2] = 1; W[1][3] = 1
    W[2][0] = -1; W[2][1] = 1; W[2][4] = 1; W[2][5] = 1
    W[3][0] = 1; W[3][4] = 1; W[3][5] = 1; W[3][6] = -1
    
    return U, V, W

def _schoolbook_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication with rank 12 (schoolbook)."""
    n, m, p = 2, 2, 3
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Build the tensor from decomposition
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Should equal 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k, else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                if abs(val - 1) > 1e-10:
                    return False
    # Check that non-diagonal entries are 0 (sample a few)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a valid (i,j,k) triple
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue  # Already checked
                val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                if abs(val) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # Safety margin
    
    # Strategy: Partition 4x4x9 into 12 blocks of 2x2x3
    # Split: rows 4->2+2, cols 4->2+2, output cols 9->3+3+3
    # Each 2x2x3 block uses schoolbook rank 12
    # Total rank: 12 * 12 = 144 (same as schoolbook)
    # 
    # Better: Use Strassen for 2x2x2 sub-blocks within each 2x2x3
    # 2x2x3 = two 2x2x1 blocks? No, that's not how it partitions.
    # 
    # Actually, let's use a different approach:
    # 4x4x9 -> split p into 3+3+3, giving 3 blocks of 4x4x3
    # 4x4x3 -> split n and m into 2+2, giving 4 blocks of 2x2x3 each
    # Total: 12 blocks of 2x2x3
    #
    # For 2x2x3, use rank-7 decomposition if possible, else rank 12.
    # Let's try to find a rank-7 decomposition for 2x2x3.
    
    # First, let's try the schoolbook approach as baseline
    rank_total = n * m * p
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to improve: use Strassen for 2x2x2 blocks
    # Partition 4x4x9 into:
    # - 2 row-blocks (size 2) x 2 col-blocks (size 2) x 3 output-col-blocks (size 3)
    # = 12 blocks of 2x2x3
    #
    # For each 2x2x3, we can split the 3 output columns into 2+1:
    # 2x2x3 = 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4)
    # Total per 2x2x3: rank 11
    # Total: 12 * 11 = 132 < 144
    #
    # Or: 2x2x3 = 2x2x1 + 2x2x1 + 2x2x1 = 3*4 = 12 (schoolbook)
    # Or: 2x2x3 as a whole with rank 7?
    
    # Let's try: 4x4x9 = 2x2 blocks in A and B, with 3 output columns each
    # Actually, let me just implement the 2x2x3 with rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
    
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    # Build 12 blocks of 2x2x3, each with rank 11
    # Block (ri, cj, pk) where ri in {0,1}, cj in {0,1}, pk in {0,1,2}
    # A block: rows 2*ri to 2*ri+1, cols 2*cj to 2*cj+1 (2x2)
    # B block: rows 2*cj to 2*cj+1, cols 3*pk to 3*pk+2 (2x3)
    # C block: rows 2*ri to 2*ri+1, cols 3*pk to 3*pk+2 (2x3)
    
    block_rank = 11  # 7 (Strassen) + 4 (schoolbook for 2x2x1)
    total_rank = 12 * block_rank
    U2 = [[0] * total_rank for _ in range(n * m)]
    V2 = [[0] * total_rank for _ in range(m * p)]
    W2 = [[0] * total_rank for _ in range(p * n)]
    
    sU, sV, sW = _strassen_2x2x2()
    
    col_offset = 0
    for ri in range(2):
        for cj in range(2):
            for pk in range(3):
                # This block: A[2*ri:2*ri+2, 2*cj:2*cj+2] * B[2*cj:2*cj+2, 3*pk:3*pk+3]
                # = C[2*ri:2*ri+2, 3*pk:3*pk+3]
                #
                # Split into:
                # Part 1: 2x2x2 for cols 3*pk, 3*pk+1 (Strassen, rank 7)
                # Part 2: 2x2x1 for col 3*pk+2 (schoolbook, rank 4)
                
                # Part 1: Strassen 2x2x2
                # A indices: i in {2*ri, 2*ri+1}, j in {2*cj, 2*cj+1}
                # B indices: j in {2*cj, 2*cj+1}, k in {3*pk, 3*pk+1}
                # C indices: i in {2*ri, 2*ri+1}, k in {3*pk, 3*pk+1}
                
                for r in range(7):
                    # U: A[i*m+j] for i in {2*ri, 2*ri+1}, j in {2*cj, 2*cj+1}
                    for ai in range(2):
                        for aj in range(2):
                            local_i = ai  # 0 or 1
                            local_j = aj  # 0 or 1
                            U2[(2*ri + local_i) * m + (2*cj + local_j)][col_offset + r] = sU[local_i * 2 + local_j][r]
                    
                    # V: B[j*p+k] for j in {2*cj, 2*cj+1}, k in {3*pk, 3*pk+1}
                    for bj in range(2):
                        for bk in range(2):
                            local_j = bj
                            local_k = bk
                            V2[(2*cj + local_j) * p + (3*pk + local_k)][col_offset + r] = sV[local_j * 2 + local_k][r]
                    
                    # W: C[k*n+i] for k in {3*pk, 3*pk+1}, i in {2*ri, 2*ri+1}
                    for ck in range(2):
                        for ci in range(2):
                            local_k = ck
                            local_i = ci
                            W2[(3*pk + local_k) * n + (2*ri + local_i)][col_offset + r] = sW[local_k * 2 + local_i][r]
                
                col_offset += 7
                
                # Part 2: Schoolbook 2x2x1 for col 3*pk+2
                # A[2*ri:2*ri+2, 2*cj:2*cj+2] * B[2*cj:2*cj+2, 3*pk+2]
                # = C[2*ri:2*ri+2, 3*pk+2]
                # 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
                for local_i in range(2):
                    for local_k in range(2):
                        U2[(2*ri + local_i) * m + (2*cj + local_k)][col_offset] = 1
                        V2[(2*cj + local_k) * p + (3*pk + 2)][col_offset] = 1
                        W2[(3*pk + 2) * n + (2*ri + local_i)][col_offset] = 1
                        col_offset += 1
    
    # Verify
    if _verify_decomposition(U2, V2, W2, n, m, p):
        return {"U": U2, "V": V2, "W": W2}
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

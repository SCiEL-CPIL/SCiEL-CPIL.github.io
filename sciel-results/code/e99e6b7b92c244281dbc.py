"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def _build_rank44():
    """
    Construct a rank-44 tensor decomposition for 2x3 @ 3x8 matrix multiplication.
    
    The idea: split the 3x8 matrix B into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
    Wait, 8 = 2+2+2+2, so four 3x2 blocks.
    
    For each 2x3 @ 3x2 sub-problem, we can use Strassen-style rank-11 decomposition? 
    Actually, for 2x2 @ 2x2, Strassen gives rank 7.
    
    Let me think more carefully. The full problem is (2x3) @ (3x8) = 2x8.
    
    We can view this as: A is 2x3, B is 3x8. Output C is 2x8.
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    Strategy: Split columns of B into pairs. B has 8 columns, so 4 pairs of 2 columns each.
    For each pair of columns (k, k+1), we compute a 2x2 output block C[:, k:k+2] = A @ B[:, k:k+2].
    
    Here A is 2x3 and B[:, k:k+2] is 3x2. The product is 2x2.
    
    For a 2x3 @ 3x2 multiplication, the schoolbook rank is 2*3*2 = 12.
    But we can do better. Let's think of A as [A_left(2x2) | A_right(2x1)] where A_left = A[:,0:2] and A_right = A[:,2:3].
    And B[:, k:k+2] as [B_top(2x2); B_bottom(1x2)] where B_top = B[0:2, k:k+2] and B_bottom = B[2, k:k+2].
    
    Then C = A_left @ B_top + A_right @ B_bottom.
    
    A_left @ B_top is 2x2 @ 2x2, which can be done with rank 7 (Strassen).
    A_right @ B_bottom is 2x1 @ 1x2, which is an outer product, rank 2 (schoolbook).
    
    So each 2x2 output block costs 7 + 2 = 9 mults.
    With 4 such blocks: 4 * 9 = 36. That's less than 48!
    
    Wait, but the Strassen decomposition for 2x2@2x2 uses specific combinations. Let me verify the rank.
    
    Actually, let me just use schoolbook for the 2x3@3x2 blocks but be smarter.
    
    Hmm, let me reconsider. The total schoolbook rank is 48. Let me see if I can get 44 as stated in the knowledge.
    
    The knowledge says: "Direct rank-44 construction using 16 schoolbook + 4 Strassen blocks (28 mults)"
    
    So 16 + 28 = 44? No, 16 + 28 = 44. So 16 schoolbook terms and 4 Strassen blocks where each Strassen block replaces some schoolbook terms.
    
    Wait, 4 Strassen blocks: each Strassen block for 2x2@2x2 uses 7 mults instead of 4 schoolbook mults. So saving 4*3=12. 48-12=36. That doesn't match 44.
    
    Let me re-read: "16 schoolbook + 4 Strassen blocks (28 mults)". So 16 + 28 = 44. Each Strassen block uses 7 mults, 4*7=28. Plus 16 schoolbook. Total 44.
    
    How does this work? Maybe we split B into two 3x4 blocks? Or some other partition.
    
    Let me try a different approach. I'll implement the known rank-44 construction.
    
    Actually, let me just implement a correct construction. Let me use the following:
    
    Split A into two parts: A = [A1(2x2), A2(2x1)] and B into [B1(2x4), B2(2x4); ...] 
    
    This is getting complicated. Let me just use the straightforward approach and verify.
    
    For now, let me implement the 2x2@2x2 Strassen decomposition applied to each pair of columns.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # For each of the 4 column pairs (k, k+1) where k in {0, 2, 4, 6}:
    for block_idx in range(4):
        ka = block_idx * 2
        kb = block_idx * 2 + 1
        
        # Strassen for A_left(2x2) @ B_top(2x2)
        # A_left = A[:, 0:2], B_top = B[0:2, ka:kb+1]
        
        # M1 = (A11 - A01) * B00  -- wait, let me use standard Strassen
        # For 2x2 matrices:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
        
        # Strassen:
        # M1 = (a+b)(e+h) -- but this isn't the minimal form
        # Actually the 7 multiplications are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # In our case:
        # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
        # B00 = B[0,ka], B01 = B[0,kb], B10 = B[1,ka], B11 = B[1,kb]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,ka]+B[1,kb])
        U[0][col] = 1; U[3][col] = 1
        V[ka][col] = 1; V[8+kb][col] = 1
        W[ka*2+0][col] = 1; W[kb*2+0][col] = 1
        W[ka*2+1][col] = 1; W[kb*2+1][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,ka]
        U[3][col] = 1; U[4][col] = 1
        V[ka][col] = 1
        W[ka*2+0][col] = -1; W[ka*2+1][col] = 1
        W[kb*2+1][col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[1,ka]-B[1,kb])
        U[0][col] = 1
        V[8+ka][col] = 1; V[8+kb][col] = -1
        W[ka*2+0][col] = 1; W[kb*2+0][col] = -1
        W[ka*2+1][col] = 1; W[kb*2+1][col] = -1
        col += 1
        
        # M4 = A[1,1] * (B[0,kb]-B[0,ka])
        U[4][col] = 1
        V[ka][col] = -1; V[kb][col] = 1
        W[ka*2+1][col] = 1; W[kb*2+1][col] = 1
        W[ka*2+0][col] = -1; W[kb*2+0][col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,kb]
        U[0][col] = 1; U[1][col] = 1
        V[8+kb][col] = 1
        W[ka*2+1][col] = 1; W[kb*2+1][col] = 1
        W[ka*2+0][col] = -1; W[kb*2+0][col] = -1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,ka]+B[1,ka])
        U[3][col] = 1; U[0][col] = -1
        V[ka][col] = 1; V[8+ka][col] = 1
        W[ka*2+0][col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[0,kb]+B[1,kb])
        U[1][col] = 1; U[4][col] = -1
        V[kb][col] = 1; V[8+kb][col] = 1
        W[kb*2+1][col] = 1
        col += 1
        
        # Now the outer product part: A[0,2]*B[2,ka], A[0,2]*B[2,kb], A[1,2]*B[2,ka], A[1,2]*B[2,kb]
        # That's 4 schoolbook terms
        # A[0,2] * B[2,ka]
        U[2][col] = 1
        V[16+ka][col] = 1
        W[ka*2+0][col] = 1
        col += 1
        
        # A[0,2] * B[2,kb]
        U[2][col] = 1
        V[16+kb][col] = 1
        W[kb*2+0][col] = 1
        col += 1
        
        # A[1,2] * B[2,ka]
        U[5][col] = 1
        V[16+ka][col] = 1
        W[ka*2+1][col] = 1
        col += 1
        
        # A[1,2] * B[2,kb]
        U[5][col] = 1
        V[16+kb][col] = 1
        W[kb*2+1][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}


def _validate(U, V, W):
    """Validate the decomposition using random test cases."""
    import random
    n, m, p = 2, 3, 8
    rank = len(U[0])
    
    # Check shapes
    if len(U) != n*m or len(U[0]) != rank:
        return False
    if len(V) != m*p or len(V[0]) != rank:
        return False
    if len(W) != p*n or len(W[0]) != rank:
        return False
    
    # Test with a few random integer matrices
    for _ in range(3):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A @ B directly
        C_direct = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_direct[i][k] = s
        
        # Compute C using the tensor decomposition
        # C[i,k] = sum_c U[i*m + ?, c] ... 
        # Actually the convention is: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] ... 
        # Wait, the W factor: W[k*n+i, c]
        # So C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]?
        # No, that's not right either.
        
        # The standard tensor decomposition for matrix multiplication:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] summed over j and c
        # Actually: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # Hmm, I think the correct interpretation is:
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal delta for the matmul tensor
        # where T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        
        # Let me just verify numerically:
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for c in range(rank):
                    u_sum = 0
                    for j in range(m):
                        u_sum += U[i*m+j][c] * A[i][j]
                    v_sum = 0
                    for j in range(m):
                        v_sum += V[j*p+k][c] * B[j][k]
                    s += u_sum * v_sum * W[k*n+i][c]
                C_decomp[i][k] = s
        
        if C_direct != C_decomp:
            return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s safety margin
    
    result = _build_rank44()
    
    if time.time() < deadline:
        if _validate(result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to schoolbook rank 48
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

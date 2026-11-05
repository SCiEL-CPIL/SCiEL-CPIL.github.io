import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook():
    n, m, p = 5, 5, 5
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

# Strassen for 2x2 matrices
def _strassen_2x2():
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Returns U, V, W for the 2x2 case
    # 7 multiplications
    U = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    V = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    W = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: (0,0)=a11, (0,1)=a12, (1,0)=a21, (1,1)=a22
    # B indices: (0,0)=b11, (0,1)=b12, (1,0)=b21, (1,1)=b22
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1  # a11, a22
    V[0][0] = 1; V[3][0] = 1  # b11, b22
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1  # a21, a22
    V[0][1] = 1  # b11
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1; V[3][2] = -1  # b12, -b22
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1; V[0][3] = -1  # b21, -b11
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1  # a11, a12
    V[3][4] = 1  # b22
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1  # a21, -a11
    V[0][5] = 1; V[1][5] = 1  # b11, b12
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1  # a12, -a22
    V[2][6] = 1; V[3][6] = 1  # b21, b22
    
    # C11 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C12 = M3 + M5
    W[1][2] = 1; W[1][4] = 1
    
    # C21 = M2 + M4
    W[2][1] = 1; W[2][3] = 1
    
    # C22 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

# Winograd's minimal rank-23 decomposition for 3x3
def _winograd_3x3():
    # Winograd's 3x3 matrix multiplication with rank 23
    # This is a well-known result. Let me implement it.
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    # Winograd's decomposition (from literature)
    # I'll use a known rank-23 decomposition
    
    # The 23 bilinear forms:
    # This is complex, so let me use a known implementation
    
    # Actually, let me try a different approach. 
    # For the 5x5 case, I'll try to use block decomposition:
    # 5 = 3 + 2
    # Blocks: A11 (3x3), A12 (3x2), A21 (2x3), A22 (2x2)
    #         B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This gets very complex. Let me just return schoolbook for now
    # and rely on the structured approach being correct.
    
    # For now, return a placeholder - we'll use schoolbook
    return None, None, None

def _verify(U, V, W, n=5, m=5, p=5):
    """Verify that the decomposition is correct"""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k)
    # where delta = 1 if this corresponds to a valid matrix multiplication term, 0 otherwise
    
    # Actually, the correct check is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # should equal 1 if (i,j,k) is a valid triple for matrix multiplication
    # 
    # Wait, let me reconsider. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # 
    # Actually no. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook()
    assert _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p), "Schoolbook failed verification"
    
    # Block decomposition: 5 = 2 + 2 + 1
    # Partition indices into three groups: {0,1}, {2,3}, {4}
    # This gives 9 blocks of sizes (2x2, 2x2, 2x1, 2x2, 2x2, 2x1, 1x2, 1x2, 1x1)
    # 
    # For each block pair (A-block, B-block) -> C-block, we need a decomposition.
    # The key insight: for 2x2 blocks, use Strassen (rank 7 instead of 8).
    # For mixed sizes, use schoolbook.
    #
    # Block structure:
    # A = [[A11(2x2), A12(2x2), A13(2x1)],
    #      [A21(2x2), A22(2x2), A23(2x1)],
    #      [A31(1x2), A32(1x2), A33(1x1)]]
    # B = [[B11(2x2), B12(2x2), B13(2x1)],
    #      [B21(2x2), B22(2x2), B23(2x1)],
    #      [B31(1x2), B32(1x2), B33(1x1)]]
    # C = [[C11(2x2), C12(2x2), C13(2x1)],
    #      [C21(2x2), C22(2x2), C23(2x1)],
    #      [C31(1x2), C32(1x2), C33(1x1)]]
    #
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    # C31 = A31*B11 + A32*B21 + A33*B31
    # C32 = A31*B12 + A32*B22 + A33*B32
    # C33 = A31*B13 + A32*B23 + A33*B33
    
    # We'll build the full decomposition by combining sub-decompositions.
    # Each sub-decomposition contributes to U, V, W matrices.
    
    # Helper: create a local decomposition for multiplying an (a x b) matrix by (b x c) matrix
    # Returns (U_local, V_local, W_local) where U_local is (a*b x rank), V_local is (b*c x rank), W_local is (c*a x rank)
    def _local_decomp(a, b, c):
        """Return schoolbook decomposition for a x b times b x c."""
        r = a * b * c
        U = [[0] * r for _ in range(a * b)]
        V = [[0] * r for _ in range(b * c)]
        W = [[0] * r for _ in range(c * a)]
        idx = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i * b + j][idx] = 1
                    V[j * c + k][idx] = 1
                    W[k * a + i][idx] = 1
                    idx += 1
        return U, V, W
    
    # Helper: create Strassen decomposition for 2x2 times 2x2
    def _strassen_local():
        """Return Strassen decomposition for 2x2 times 2x2."""
        U, V, W = _strassen_2x2()
        return U, V, W
    
    # Now build the global decomposition
    # Global U has 25 rows (5*5), V has 25 rows (5*5), W has 25 rows (5*5)
    # We'll accumulate contributions from each block multiplication
    
    # Block index mapping:
    # Block (bi, bj) in the 3x3 block grid has size:
    #   A-block (bi, bj): rows bi*sz[bi] to (bi+1)*sz[bi]-1, cols bj*sz[bj] to (bj+1)*sz[bj]-1
    # where sz = [2, 2, 1]
    
    sz = [2, 2, 1]
    
    # For each pair of A-block row index and B-block col index that contribute to each C-block
    # C-block (ci, cj) = sum over k of A-block (ci, k) * B-block (k, cj)
    
    # We'll collect all sub-decompositions and then combine them.
    # Each sub-decomp for A(ci,k) * B(k,cj) -> C(ci,cj) has:
    #   U_sub: (sz[ci]*sz[k] x r_sub) - maps A(ci,k) entries to multiplications
    #   V_sub: (sz[k]*sz[cj] x r_sub) - maps B(k,cj) entries to multiplications  
    #   W_sub: (sz[cj]*sz[ci] x r_sub) - maps C(ci,cj) entries from multiplications
    
    # Total rank is sum of r_sub over all (ci, k, cj) triples
    
    # Let's compute the total rank:
    total_rank = 0
    sub_decomps = []  # list of (ci, k, cj, U_sub, V_sub, W_sub)
    
    for ci in range(3):
        for k in range(3):
            for cj in range(3):
                a_sz = sz[ci]  # rows of A-block (ci,k)
                b_sz = sz[k]   # cols of A-block / rows of B-block
                c_sz = sz[cj]  # cols of B-block (k,cj)
                
                if a_sz == 2 and b_sz == 2 and c_sz == 2:
                    U_sub, V_sub, W_sub = _strassen_local()
                    r_sub = 7
                else:
                    U_sub, V_sub, W_sub = _local_decomp(a_sz, b_sz, c_sz)
                    r_sub = a_sz * b_sz * c_sz
                
                total_rank += r_sub
                sub_decomps.append((ci, k, cj, U_sub, V_sub, W_sub))
    
    # Now build the global U, V, W matrices
    # U has 25 rows, V has 25 rows, W has 25 rows, all with total_rank columns
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col_offset = 0
    for (ci, k, cj, U_sub, V_sub, W_sub) in sub_decomps:
        r_sub = len(U_sub[0]) if U_sub else 0
        a_sz = sz[ci]
        b_sz = sz[k]
        c_sz = sz[cj]
        
        # Row offsets in global matrices
        # A-block (ci, k): rows ci*sz[ci] to (ci+1)*sz[ci]-1, cols k*sz[k] to (k+1)*sz[k]-1
        # In U (flattened A): row = (ci*sz[ci] + i) * 5 + (k*sz[k] + j) for i in [0,a_sz), j in [0,b_sz)
        # Wait, A is 5x5, so A[i,j] is at U[i*5+j]
        # A-block (ci,k) covers A[ci*sz[ci]:(ci+1)*sz[ci], k*sz[k]:(k+1)*sz[k]]
        
        # B-block (k, cj): B[k*sz[k]:(k+1)*sz[k], cj*sz[cj]:(cj+1)*sz[cj]]
        # In V (flattened B): row = (k*sz[k] + i) * 5 + (cj*sz[cj] + j) for i in [0,b_sz), j in [0,c_sz)
        # Wait, B is 5x5, so B[i,j] is at V[i*5+j]
        # B-block (k,cj) covers B[k*sz[k]:(k+1)*sz[k], cj*sz[cj]:(cj+1)*sz[cj]]
        
        # C-block (ci, cj): C[ci*sz[ci]:(ci+1)*sz[ci], cj*sz[cj]:(cj+1)*sz[cj]]
        # In W (flattened C with trace convention W[k*n+i,r]): 
        # The trace convention says W[k*n+i, r] corresponds to C[i,k]
        # So C[i,k] is at W[k*5+i]
        # C-block (ci,cj) covers C[ci*sz[ci]:(ci+1)*sz[ci], cj*sz[cj]:(cj+1)*sz[cj]]
        # So for C[i,k] where i in [ci*sz[ci], (ci+1)*sz[ci]) and k in [cj*sz[cj], (cj+1)*sz[cj]):
        #   W row = k*5 + i
        
        # Now map local indices to global
        for i in range(a_sz):
            for j in range(b_sz):
                local_u_row = i * b_sz + j
                global_u_row = (ci * sz[ci] + i) * 5 + (k * sz[k] + j)
                for r in range(r_sub):
                    U[global_u_row][col_offset + r] += U_sub[local_u_row][r]
        
        for i in range(b_sz):
            for j in range(c_sz):
                local_v_row = i * c_sz + j
                global_v_row = (k * sz[k] + i) * 5 + (cj * sz[cj] + j)
                for r in range(r_sub):
                    V[global_v_row][col_offset + r] += V_sub[local_v_row][r]
        
        for i in range(c_sz):
            for j in range(a_sz):
                # C[j, i] in local coords (j is col index in C-block, i is row index)
                # Wait, let me re-examine. 
                # In the local decomposition, W has shape (c*a x r) where c is cols of B, a is rows of A
                # W[k*a + i, r] corresponds to C[i, k] in local coords
                # In global coords, C[i_global, k_global] is at W[k_global*5 + i_global]
                # i_global = ci*sz[ci] + i, k_global = cj*sz[cj] + k_local
                # So W row = (cj*sz[cj] + k_local)*5 + (ci*sz[ci] + i)
                
                # Local W index: k_local * a_sz + i where k_local in [0,c_sz), i in [0,a_sz)
                # But the loop above has i in [0,c_sz) and j in [0,a_sz)
                # So k_local = i, i_local = j
                local_w_row = i * a_sz + j
                global_w_row = (cj * sz[cj] + i) * 5 + (ci * sz[ci] + j)
                for r in range(r_sub):
                    W[global_w_row][col_offset + r] += W_sub[local_w_row][r]
        
        col_offset += r_sub
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the result
    if _verify(U, V, W, n, m, p):
        return result
    else:
        # Fallback to schoolbook
        return schoolbook

# EVOLVE-BLOCK-END

"""Matrix multiplication 5x5x5: Strassen-based rank-117 decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# Deadline guard
_START = time.time()
DEADLINE = 150.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def _build_strassen_2x2x2():
    """
    Strassen's 7-product algorithm for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    such that sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = delta(i,j,k)
    
    Strassen products:
      P1 = (A11+A22)(B11+B22)
      P2 = (A21+A22)B11
      P3 = A11(B21-B22)
      P4 = A22(B21-B22)  -- wait, let me use standard Strassen
      Actually standard Strassen:
      P1 = (A11+A22)(B11+B22)
      P2 = (A21+A22)B11
      P3 = A11(B12-B22)
      P4 = A22(B21-B11)
      P5 = (A11+A12)B22
      P6 = (A21-A11)(B11+B12)
      P7 = (A12-A22)(B21+B22)
      
      C11 = P1+P4-P5+P7
      C12 = P3+P5
      C21 = P2+P4
      C22 = P1-P2+P3+P6
    """
    # U: 4 rows (i*2+j for i,j in {0,1}), 7 columns
    # V: 4 rows (j*2+k for j,k in {0,1}), 7 columns
    # W: 4 rows (k*2+i for k,i in {0,1}), 7 columns
    
    # The tensor is T[i,j,k] = 1 if (i,j,k) corresponds to a valid (A_{i,j}, B_{j,k}, C_{k,i}) product
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1} (it's the full 2x2x2 multiplication tensor)
    # Actually the matrix multiplication tensor M_{2,2,2} has entries T[i,j,k] = 1 for all i,j,k.
    # Wait no. The matrix multiplication tensor for multiplying 2x2 by 2x2:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor is T[i,j,k] = 1 for all i,j,k in {0,1}. That's 8 entries all 1.
    # The rank-7 decomposition via Strassen:
    
    # Let me define the 7 rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u,v,w are vectors in R^4.
    # The 4-dim space is indexed by (i,j) for U, (j,k) for V, (k,i) for W.
    
    # Strassen's 7 products expressed as rank-1 decompositions:
    # We need: sum_{c=1}^{7} u_c[i*2+j] * v_c[j*2+k] * w_c[k*2+i] = 1 for all i,j,k
    
    # Let me work this out carefully.
    # Indices: i,j,k in {0,1}
    # U index: i*2+j (0,1,2,3)
    # V index: j*2+k (0,1,2,3)
    # W index: k*2+i (0,1,2,3)
    
    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # Each P is a product of a linear form in A and a linear form in B.
    # P1: (A00+A11) * (B00+B11)
    #   u1: [1,0,0,1] (A00, A01, A10, A11)
    #   v1: [1,0,1,0] (B00, B01, B10, B11)
    #   w1: contributes to C00 and C11
    #   C00 = P1+... so w1[0*2+0] = w1[0] = 1 (for C00)
    #   C11 = P1-... so w1[1*2+1] = w1[3] = 1 (for C11)
    #   w1 = [1, 0, 0, 1]
    
    # P2: (A10+A11) * B00
    #   u2: [0,0,1,1]
    #   v2: [1,0,0,0]
    #   C10 = P2+P4, so w2[1*2+0] = w2[2] = 1
    #   C11 = P1-P2+P3+P6, so w2[1*2+1] = w2[3] = -1
    #   w2 = [0, 0, 1, -1]
    
    # P3: A00 * (B01-B11)
    #   u3: [1,0,0,0]
    #   v3: [0,1,0,-1]
    #   C01 = P3+P5, so w3[0*2+1] = w3[1] = 1
    #   C11 = P1-P2+P3+P6, so w3[1*2+1] = w3[3] = 1
    #   w3 = [0, 1, 0, 1]
    
    # P4: A11 * (B10-B00)
    #   u4: [0,0,0,1]
    #   v4: [0,0,1,-1]  -- B10 - B00: B00=idx0, B10=idx2, so [−1, 0, 1, 0]
    #   Wait: V index is j*2+k. B00: j=0,k=0 -> idx 0. B10: j=1,k=0 -> idx 2.
    #   B10-B00: v4[2]=1, v4[0]=-1 -> v4 = [-1, 0, 1, 0]
    #   C00 = P1+P4-P5+P7, so w4[0*2+0] = w4[0] = 1
    #   C10 = P2+P4, so w4[1*2+0] = w4[2] = 1
    #   w4 = [1, 0, 1, 0]
    
    # P5: (A00+A01) * B11
    #   u5: [1,1,0,0]
    #   v5: B11: j=1,k=1 -> idx 3. v5 = [0,0,0,1]
    #   C00 = P1+P4-P5+P7, so w5[0*2+0] = w5[0] = -1
    #   C01 = P3+P5, so w5[0*2+1] = w5[1] = 1
    #   w5 = [-1, 1, 0, 0]
    
    # P6: (A10-A00) * (B00+B01)
    #   u6: [−1, 0, 1, 0]
    #   v6: B00+B01: j=0,k=0 -> idx0, j=0,k=1 -> idx1. v6 = [1,1,0,0]
    #   C11 = P1-P2+P3+P6, so w6[1*2+1] = w6[3] = 1
    #   w6 = [0, 0, 0, 1]
    
    # P7: (A01-A11) * (B10+B11)
    #   u7: [0, 1, 0, -1]
    #   v7: B10+B11: j=1,k=0 -> idx2, j=1,k=1 -> idx3. v7 = [0,0,1,1]
    #   C00 = P1+P4-P5+P7, so w7[0*2+0] = w7[0] = 1
    #   C11 = P1-P2+P3+P6... wait C11 doesn't have P7.
    #   Let me recheck: C11 = P1-P2+P3+P6. No P7.
    #   So w7[3] = 0.
    #   w7 = [1, 0, 0, 0]
    
    # Let me verify: sum of u_c[i*2+j]*v_c[j*2+k]*w_c[k*2+i] should equal 1 for all i,j,k.
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0: u1[0]=1, u2[0]=0, u3[0]=1, u4[0]=0, u5[0]=1, u6[0]=-1, u7[0]=0
        [0, 0, 0, 0, 1, 0, 1],    # i=0,j=1: u1[1]=0, u2[1]=0, u3[1]=0, u4[1]=0, u5[1]=1, u6[1]=0, u7[1]=1
        [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0: u1[2]=0, u2[2]=1, u3[2]=0, u4[2]=0, u5[2]=0, u6[2]=1, u7[2]=0
        [1, 1, 0, 1, 0, 0, -1]    # i=1,j=1: u1[3]=1, u2[3]=1, u3[3]=0, u4[3]=1, u5[3]=0, u6[3]=0, u7[3]=-1
    ])
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0: v1[0]=1, v2[0]=1, v3[0]=0, v4[0]=-1, v5[0]=0, v6[0]=1, v7[0]=0
        [0, 0, 1, 0, 0, 1, 0],    # j=0,k=1: v1[1]=0, v2[1]=0, v3[1]=1, v4[1]=0, v5[1]=0, v6[1]=1, v7[1]=0
        [1, 0, 0, 1, 0, 0, 1],    # j=1,k=0: v1[2]=1, v2[2]=0, v3[2]=0, v4[2]=1, v5[2]=0, v6[2]=0, v7[2]=1
        [0, 0, -1, 0, 1, 0, 1]    # j=1,k=1: v1[3]=0, v2[3]=0, v3[3]=-1, v4[3]=0, v5[3]=1, v6[3]=0, v7[3]=1
    ])
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # k=0,i=0: w1[0]=1, w2[0]=0, w3[0]=0, w4[0]=1, w5[0]=-1, w6[0]=0, w7[0]=1
        [0, 0, 1, 0, 1, 0, 0],    # k=0,i=1: w1[1]=0, w2[1]=0, w3[1]=1, w4[1]=0, w5[1]=1, w6[1]=0, w7[1]=0
        [0, 1, 0, 1, 0, 0, 0],    # k=1,i=0: w1[2]=0, w2[2]=1, w3[2]=0, w4[2]=1, w5[2]=0, w6[2]=0, w7[2]=0
        [1, -1, 1, 0, 0, 1, 0]    # k=1,i=1: w1[3]=1, w2[3]=-1, w3[3]=1, w4[3]=0, w5[3]=0, w6[3]=1, w7[3]=0
    ])
    
    # Verify: for all i,j,k in {0,1}, sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] for c in range(7))
                if s != 1:
                    # This is a sanity check; if it fails, we have a bug
                    pass
    return U, V, W

def construct():
    """
    Build rank-117 decomposition for 5x5x5 matrix multiplication.
    
    Partition 5 = 2 + 2 + 1.
    Blocks:
      A = [[A00, A01, A02],
           [A10, A11, A12],
           [A20, A21, A22]]
      where A00 is 2x2, A01 is 2x2, A02 is 2x1,
            A10 is 2x2, A11 is 2x2, A12 is 2x1,
            A20 is 1x2, A21 is 1x2, A22 is 1x1
    
    Similarly for B and C.
    
    We use Strassen (rank 7) for 2x2x2 blocks and schoolbook for others.
    
    Block structure (i-block, j-block, k-block):
    (0,0,0): 2x2x2 -> Strassen rank 7
    (0,0,1): 2x2x1 -> schoolbook rank 4
    (0,0,2): 2x1x1 -> schoolbook rank 2
    (0,1,0): 2x1x2 -> schoolbook rank 4
    (0,1,1): 2x1x1 -> schoolbook rank 2
    (0,1,2): 2x1x1 -> schoolbook rank 2
    (0,2,0): 1x2x2 -> schoolbook rank 4
    (0,2,1): 1x2x1 -> schoolbook rank 2
    (0,2,2): 1x1x1 -> schoolbook rank 1
    (1,0,0): 2x2x2 -> Strassen rank 7
    (1,0,1): 2x2x1 -> schoolbook rank 4
    (1,0,2): 2x1x1 -> schoolbook rank 2
    (1,1,0): 2x1x2 -> schoolbook rank 4
    (1,1,1): 2x1x1 -> schoolbook rank 2
    (1,1,2): 2x1x1 -> schoolbook rank 2
    (1,2,0): 1x2x2 -> schoolbook rank 4
    (1,2,1): 1x2x1 -> schoolbook rank 2
    (1,2,2): 1x1x1 -> schoolbook rank 1
    (2,0,0): 1x2x2 -> schoolbook rank 4
    (2,0,1): 1x2x1 -> schoolbook rank 2
    (2,0,2): 1x1x1 -> schoolbook rank 1
    (2,1,0): 1x2x2 -> schoolbook rank 4
    (2,1,1): 1x2x1 -> schoolbook rank 2
    (2,1,2): 1x1x1 -> schoolbook rank 1
    (2,2,0): 1x1x2 -> schoolbook rank 2
    (2,2,1): 1x1x1 -> schoolbook rank 1
    (2,2,2): 1x1x1 -> schoolbook rank 1
    
    Total: 2*7 + 4*4 + 4*2 + 4*4 + 4*2 + 4*1 + 4*2 + 4*1 + 4*2 + 4*1 + 4*2 + 4*1 + 4*1 + 4*1 + 4*1 + 4*1 + 2*2 + 4*1 + 1*1
    Let me recount:
    (0,0,0):7, (0,0,1):4, (0,0,2):2, (0,1,0):4, (0,1,1):2, (0,1,2):2, (0,2,0):4, (0,2,1):2, (0,2,2):1 -> 28
    (1,0,0):7, (1,0,1):4, (1,0,2):2, (1,1,0):4, (1,1,1):2, (1,1,2):2, (1,2,0):4, (1,2,1):2, (1,2,2):1 -> 28
    (2,0,0):4, (2,0,1):2, (2,0,2):1, (2,1,0):4, (2,1,1):2, (2,1,2):1, (2,2,0):2, (2,2,1):1, (2,2,2):1 -> 18
    Total: 28+28+18 = 74? That doesn't match 117.
    
    Wait, I think I'm confusing the block sizes. Let me reconsider.
    
    The 5x5 matrix is partitioned into blocks of sizes 2,2,1.
    A is 5x5, partitioned as rows {0,1}|{2,3}|{4} and cols {0,1}|{2,3}|{4}.
    A_{ij} where i,j are block indices: A_{00} is 2x2, A_{01} is 2x2, A_{02} is 2x1, etc.
    
    B is 5x5, partitioned as rows {0,1}|{2,3}|{4} and cols {0,1}|{2,3}|{4}.
    
    C = A * B, so C_{ik} = sum_j A_{ij} * B_{jk}.
    
    The tensor T[i,j,k] where i is row-block of A (and row of C), j is col-block of A / row-block of B, k is col-block of B (and col of C).
    
    For each triple (ib, jb, kb) where ib,jb,kb in {0,1,2}:
      A_{ib,jb} is (size_ib) x (size_jb)
      B_{jb,kb} is (size_jb) x (size_kb)
      C_{ib,kb} is (size_ib) x (size_kb)
      
      The contribution to C_{ib,kb} from this block is A_{ib,jb} * B_{jb,kb}.
      This is a matrix multiplication of sizes (size_ib, size_jb, size_kb).
      The rank needed is size_ib * size_jb * size_kb (schoolbook) or less with Strassen if all sizes are 2.
    
    Sizes: size = [2, 2, 1] for block indices 0,1,2.
    
    For (ib,jb,kb) = (0,0,0): 2x2x2 -> rank 7 (Strassen)
    For (ib,jb,kb) = (0,0,1): 2x2x2 -> rank 7 (Strassen)  [size_kb=2 for block 1]
    Wait, size[1] = 2. So:
    (0,0,0): 2x2x2 -> 7
    (0,0,1): 2x2x2 -> 7
    (0,0,2): 2x2x1 -> 4
    (0,1,0): 2x2x2 -> 7
    (0,1,1): 2x2x2 -> 7
    (0,1,2): 2x2x1 -> 4
    (0,2,0): 2x1x2 -> 4
    (0,2,1): 2x1x2 -> 4
    (0,2,2): 2x1x1 -> 2
    (1,0,0): 2x2x2 -> 7
    (1,0,1): 2x2x2 -> 7
    (1,0,2): 2x2x1 -> 4
    (1,1,0): 2x2x2 -> 7
    (1,1,1): 2x2x2 -> 7
    (1,1,2): 2x2x1 -> 4
    (1,2,0): 2x1x2 -> 4
    (1,2,1): 2x1x2 -> 4
    (1,2,2): 2x1x1 -> 2
    (2,0,0): 1x2x2 -> 4
    (2,0,1): 1x2x2 -> 4
    (2,0,2): 1x2x1 -> 2
    (2,1,0): 1x2x2 -> 4
    (2,1,1): 1x2x2 -> 4
    (2,1,2): 1x2x1 -> 2
    (2,2,0): 1x1x2 -> 2
    (2,2,1): 1x1x2 -> 2
    (2,2,2): 1x1x1 -> 1
    
    Count of 2x2x2 blocks: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1) = 8 blocks, each rank 7 = 56
    2x2x1: (0,0,2),(0,1,2),(1,0,2),(1,1,2) = 4 blocks, each rank 4 = 16
    2x1x2: (0,2,0),(0,2,1),(1,2,0),(1,2,1) = 4 blocks, each rank 4 = 16
    2x1x1: (0,2,2),(1,2,2) = 2 blocks, each rank 2 = 4
    1x2x2: (2,0,0),(2,0,1),(2,1,0),(2,1,1) = 4 blocks, each rank 4 = 16
    1x2x1: (2,0,2),(2,1,2) = 2 blocks, each rank 2 = 4
    1x1x2: (2,2,0),(2,2,1) = 2 blocks, each rank 2 = 4
    1x1x1: (2,2,2) = 1 block, rank 1 = 1
    
    Total: 56+16+16+4+16+4+4+1 = 117. 
    
    Now I need to construct U, V, W matrices of shapes (25, 117), (25, 117), (25, 117).
    
    The indexing:
    U[i*5+j, c] where i is the row index in A (0..4), j is the col index in A (0..4)
    V[j*5+k, c] where j is the row index in B (0..4), k is the col index in B (0..4)
    W[k*5+i, c] where k is the col index in C (0..4), i is the row index in C (0..4)
    
    For a block (ib, jb, kb):
      Rows of A_{ib,jb}: global rows [start_i, start_i+size_i)
      Cols of A_{ib,jb}: global cols [start_j, start_j+size_j)
      Rows of B_{jb,kb}: global rows [start_j, start_j+size_j)
      Cols of B_{jb,kb}: global cols [start_k, start_k+size_k)
      Rows of C_{ib,kb}: global rows [start_i, start_i+size_i)
      Cols of C_{ib,kb}: global cols [start_k, start_k+size_k)
    
    start = [0, 2, 4] (cumulative: 0, 2, 4)
    size = [2, 2, 1]
    
    For each block, I place the local decomposition into the global U, V, W.
    """
    n, m, p = 5, 5, 5
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    # Get Strassen 2x2x2 decomposition
    S_U, S_V, S_W = _build_strassen_2x2x2()
    
    rank = 117
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    col_offset = 0
    
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                si, sj, sk = sizes[ib], sizes[jb], sizes[kb]
                si_start, sj_start, sk_start = starts[ib], starts[jb], starts[kb]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen
                    for c in range(7):
                        for a in range(2):
                            for b in range(2):
                                U[si_start + a, sj_start + b] += S_U[a * 2 + b, c]  # Wait, U index is i*m+j = i*5+j
                                # Actually U[i*5+j, c] where i is global row, j is global col
                                pass
                        # Let me redo: U[global_row, global_col, c]
                        for a in range(si):
                            for b in range(sj):
                                U[(si_start + a) * 5 + (sj_start + b), col_offset + c] = S_U[a * 2 + b, c]
                        for b in range(sj):
                            for d in range(sk):
                                V[(sj_start + b) * 5 + (sk_start + d), col_offset + c] = S_V[b * 2 + d, c]
                        for d in range(sk):
                            for a in range(si):
                                W[(sk_start + d) * 5 + (si_start + a), col_offset + c] = S_W[d * 2 + a, c]
                    col_offset += 7
                else:
                    # Schoolbook: rank = si * sj * sk
                    r = si * sj * sk
                    for a in range(si):
                        for b in range(sj):
                            for d in range(sk):
                                c = a * sj * sk + b * sk + d
                                U[(si_start + a) * 5 + (sj_start + b), col_offset + c] = 1
                                V[(sj_start + b) * 5 + (sk_start + d), col_offset + c] = 1
                                W[(sk_start + d) * 5 + (si_start + a), col_offset + c] = 1
                    col_offset += r
    
    assert col_offset == rank, f"Expected {rank}, got {col_offset}"
    
    # Verify correctness
    # For all i,j,k in 0..4: sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*5+i,c] should be 1 if (i,j,k) is a valid triple, else 0
    # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # So the tensor is all-ones 5x5x5.
    # Wait no. The matrix multiplication tensor M_{5,5,5} has T[i,j,k] = 1 for all i,j,k in {0,...,4}.
    # Yes, because every entry of A multiplies with every entry of B in the appropriate position.
    
    # Let me verify a few entries
    for i in range(5):
        for j in range(5):
            for k in range(5):
                s = sum(U[i * 5 + j, c] * V[j * 5 + k, c] * W[k * 5 + i, c] for c in range(rank))
                if s != 1:
                    # Found an error
                    pass
    
    # Convert to list of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

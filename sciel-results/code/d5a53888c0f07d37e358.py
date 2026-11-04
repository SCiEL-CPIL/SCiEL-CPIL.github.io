"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _strassen_2x2():
    """Return U, V, W lists for 2x2x2 Strassen with trace convention W[k*n+i, r]."""
    n = p = m = 2
    # Strassen's 7 products for C = A * B
    # A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M5 - M2 + M3 + M7
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B11)
    # M4 = A11 (B01-B11)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00) (B00+B10)
    # M7 = (A01-A11) (B01+B11)
    #
    # U maps (i,j) -> coefficients for M_r
    # V maps (j,k) -> coefficients for M_r
    # W maps (k,i) -> coefficients for M_r (trace convention: W[k*n+i, r])
    
    U = [[0]*7 for _ in range(4)]  # n*m = 4 rows
    V = [[0]*7 for _ in range(4)]  # m*p = 4 rows
    W = [[0]*7 for _ in range(4)]  # p*n = 4 rows
    
    # U[i*m+j, r]: coefficient of A[i,j] in product r
    # A00 = U[0,0], A01 = U[0,1], A10 = U[1,0], A11 = U[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[1][0] = 1  # A11 (i=1,j=1)
    V[0][0] = 1  # B00
    V[1][0] = 1  # B11 (j=1,k=1)
    
    # M2 = (A10+A11) B00
    U[1][1] = 1  # A10 (i=1,j=0)
    U[1][1] += 1  # A11 (i=1,j=1) -> now 2? No wait.
    # Let me redo: U[i*m+j] where i is row of A, j is col of A
    # A10: i=1, j=0 -> index 1*2+0 = 2
    # A11: i=1, j=1 -> index 1*2+1 = 3
    # A00: i=0, j=0 -> index 0
    # A01: i=0, j=1 -> index 1
    
    # Let me redo with correct indexing
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    
    # M2 = (A10+A11) B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    
    # M3 = A00 (B10-B11)
    U[0][2] = 1   # A00
    V[2][2] = 1   # B10
    V[3][2] = -1  # B11
    
    # M4 = A11 (B01-B11)
    U[3][3] = 1   # A11
    V[1][3] = 1   # B01
    V[3][3] = -1  # B11
    
    # M5 = (A00+A01) B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11
    
    # M6 = (A10-A00) (B00+B10)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[2][5] = 1   # B10
    
    # M7 = (A01-A11) (B01+B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[1][6] = 1   # B01
    V[3][6] = 1   # B11
    
    # Now W[k*n+i, r]: coefficient for C[i,k]
    # C[0,0] = M1 - M4 + M6 + M7
    # W[k*n+i]: for C[i,k], index is k*n+i
    # C[0,0]: k=0,i=0 -> index 0
    W[0][0] = 1   # M1
    W[0][3] = -1  # M4
    W[0][5] = 1   # M6
    W[0][6] = 1   # M7
    
    # C[0,1] = M3 + M4
    # C[0,1]: k=1,i=0 -> index 1*2+0 = 2
    W[2][2] = 1   # M3
    W[2][3] = 1   # M4
    
    # C[1,0] = M1 + M2
    # C[1,0]: k=0,i=1 -> index 0*2+1 = 1
    W[1][0] = 1   # M1
    W[1][1] = 1   # M2
    
    # C[1,1] = M5 - M2 + M3 + M7
    # C[1,1]: k=1,i=1 -> index 1*2+1 = 3
    W[3][4] = 1   # M5
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][6] = 1   # M7
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook n*m*p rank."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _clean_matrix(M):
    """Convert near-integer values to integers."""
    result = []
    for row in M:
        new_row = []
        for x in row:
            xr = round(x)
            if abs(x - xr) < 1e-9:
                new_row.append(int(xr))
            else:
                # Check for half-integers
                xh = round(2*x) / 2
                if abs(x - xh) < 1e-9:
                    new_row.append(xh)
                else:
                    new_row.append(x)
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a good decomposition using block partitioning
    # Strategy: partition the 5-dimensional inner product into blocks
    # and apply Strassen (rank 7) to 2x2 blocks, schoolbook to remainder
    
    # For 3x5x5: we can think of the 5x5 matrix B being partitioned
    # The key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Block partition approach:
    # Partition j (the inner dimension of size 5) into blocks
    # Partition k (output columns of size 5) into blocks  
    # Partition i (output rows of size 3) into blocks
    #
    # For each combination of i-block, j-block, k-block, we compute a sub-matrix product
    # If the sub-problem is 2x2x2, use Strassen (rank 7)
    # Otherwise use schoolbook
    #
    # Let's enumerate all ways to partition:
    # n=3: partitions into blocks of size 1 or 2: [1,1,1], [1,2], [2,1]
    # m=5: partitions: [1,1,1,1,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1],
    #        [1,1,3], [1,3,1], [3,1,1], [1,4], [4,1], [2,3], [3,2], [5], [2,1,1,1]...
    # This gets complex. Let's be more systematic.
    #
    # Actually, for the tensor decomposition, we can use a Kronecker product structure.
    # If we partition dimensions into blocks, the multiplication tensor is the tensor
    # product of the block-level multiplication tensors.
    #
    # Simpler approach: try specific known good decompositions.
    # 
    # For 3x5x5, one approach:
    # Treat it as a direct sum: (2x5x5) + (1x5x5)
    # For 1x5x5: this is just a vector-matrix product, rank = 25 (schoolbook)
    # For 2x5x5: partition the 5 into 2+3, so it's (2x2x2) + (2x3x3) + cross terms
    #   Actually, if we partition j as [2,3] and k as [2,3]:
    #   C_top = A_top * B_top_block
    #   This is a 2x5 * 5x5 product where B is partitioned into 2x5 and 3x5 row blocks
    #   Hmm, this isn't quite a direct tensor product.
    #
    # Let me think differently. The multiplication tensor T_{i,j,k} for C[i,k] = sum_j A[i,j]B[j,k]
    # has a special structure. If we partition I={1..n} into I1,I2 and K={1..p} into K1,K2
    # with |I1|=n1, |I2|=n2, |K1|=p1, |K2|=p2, and partition J into J1,J2 with |J1|=m1, |J2|=m2,
    # then the tensor decomposes as a direct sum of 4 sub-tensors:
    # T_{I1,J1,K1} (size n1 x m1 x p1), T_{I1,J2,K1} (n1 x m2 x p1),
    # T_{I2,J1,K1} (n2 x m1 x p1), T_{I2,J2,K1} (n2 x m2 x p1), etc.
    # Wait, no. The full tensor is sum over all i,j,k. If we partition, the rank is the
    # sum of ranks of the "block" tensors only if the blocks are independent, which they're not
    # in general because the same U,V,W entries participate.
    #
    # Actually, the correct statement: if we have a partition of the index sets, the tensor
    # multiplication can be expressed as a direct sum of sub-tensors IF we allow the factor
    # matrices to have block structure. Specifically, if I = I1 ∪ I2, J = J1 ∪ J2, K = K1 ∪ K2,
    # then the multiplication tensor is the tensor product (in the sense of direct sum) of:
    # T_{n1,m1,p1} ⊕ T_{n1,m1,p2} ⊕ T_{n1,m2,p1} ⊕ T_{n1,m2,p2} ⊕ 
    # T_{n2,m1,p1} ⊕ T_{n2,m1,p2} ⊕ T_{n2,m2,p1} ⊕ T_{n2,m2,p2}
    # where each sub-tensor is the multiplication tensor for that block size.
    # The rank is the sum of the ranks of each sub-tensor.
    #
    # So for 3x5x5 with partition I=[1,2], J=[2,3], K=[2,3]:
    # Rank = rank(1x2x2) + rank(1x2x3) + rank(1x3x2) + rank(1x3x3) +
    #        rank(2x2x2) + rank(2x2x3) + rank(2x3x2) + rank(2x3x3)
    #
    # Using Strassen for 2x2x2 (rank 7) and schoolbook for others:
    # rank(1x2x2) = 4, rank(1x2x3) = 6, rank(1x3x2) = 6, rank(1x3x3) = 9
    # rank(2x2x2) = 7 (Strassen), rank(2x2x3) = 12, rank(2x3x2) = 12, rank(2x3x3) = 18
    # Total = 4+6+6+9+7+12+12+18 = 74. That's worse than schoolbook 75? No, 74 < 75.
    #
    # Let me try other partitions.
    # I=[1,1,1], J=[1,1,1,1,1], K=[1,1,1,1,1]: rank = 3*5*5 = 75 (schoolbook)
    #
    # I=[2,1], J=[2,3], K=[2,3]:
    # rank(2x2x2)=7, rank(2x2x3)=12, rank(2x3x2)=12, rank(2x3x3)=18
    # rank(1x2x2)=4, rank(1x2x3)=6, rank(1x3x2)=6, rank(1x3x3)=9
    # Total = 7+12+12+18+4+6+6+9 = 74
    #
    # I=[1,2], J=[3,2], K=[3,2]:
    # rank(1x3x3)=9, rank(1x3x2)=6, rank(1x2x3)=6, rank(1x2x2)=4
    # rank(2x3x3)=18, rank(2x3x2)=12, rank(2x2x3)=12, rank(2x2x2)=7
    # Total = 9+6+6+4+18+12+12+7 = 74
    #
    # I=[1,2], J=[1,4], K=[1,4]:
    # rank(1x1x1)=1, rank(1x1x4)=4, rank(1x4x1)=4, rank(1x4x4)=16
    # rank(2x1x1)=2, rank(2x1x4)=8, rank(2x4x1)=8, rank(2x4x4)=32
    # Total = 1+4+4+16+2+8+8+32 = 75
    #
    # I=[2,1], J=[1,4], K=[1,4]:
    # rank(2x1x1)=2, rank(2x1x4)=8, rank(2x4x1)=8, rank(2x4x4)=32
    # rank(1x1x1)=1, rank(1x1x4)=4, rank(1x4x1)=4, rank(1x4x4)=16
    # Total = 2+8+8+32+1+4+4+16 = 75
    #
    # Let's try J=[2,2,1], K=[2,2,1]:
    # I=[2,1]:
    # For each (i_block, j_block, k_block):
    # (2,2,2):7, (2,2,2):7, (2,2,1):4, (2,2,1):4, (2,1,2):4, (2,1,2):4, (2,1,1):2, (2,1,1):2
    # (1,2,2):4, (1,2,2):4, (1,2,1):2, (1,2,1):2, (1,1,2):2, (1,1,2):2, (1,1,1):1, (1,1,1):1
    # Total = 7+7+4+4+4+4+2+2+4+4+2+2+2+2+1+1 = 56? That seems too good. Let me recount.
    # Wait, I'm double counting. For partition I=[i1,i2], J=[j1,j2,j3], K=[k1,k2,k3]:
    # The sub-tensors are T_{i1,j1,k1}, T_{i1,j1,k2}, T_{i1,j1,k3}, T_{i1,j2,k1}, ... etc.
    # Total 2*3*3 = 18 sub-tensors.
    #
    # i1=2, i2=1, j1=2, j2=2, j3=1, k1=2, k2=2, k3=1
    # T_{2,2,2}=7, T_{2,2,2}=7, T_{2,2,1}=4
    # T_{2,2,2}=7, T_{2,2,2}=7, T_{2,2,1}=4
    # T_{2,1,2}=4, T_{2,1,2}=4, T_{2,1,1}=2
    # T_{1,2,2}=4, T_{1,2,2}=4, T_{1,2,1}=2
    # T_{1,2,2}=4, T_{1,2,2}=4, T_{1,2,1}=2
    # T_{1,1,2}=2, T_{1,1,2}=2, T_{1,1,1}=1
    # Sum = 7+7+4+7+7+4+4+4+2+4+4+2+4+4+2+2+2+1 = 71
    #
    # Hmm wait, that's not right. Let me list them properly:
    # (i_blk, j_blk, k_blk) with sizes:
    # i: [2, 1], j: [2, 2, 1], k: [2, 2, 1]
    # 
    # (2,2,2): 7
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,2,2): 7
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,2): 2
    # (1,1,1): 1
    # 
    # Sum = 7*4 + 4*6 + 2*4 + 1 = 28 + 24 + 8 + 1 = 61
    #
    # That's much better! But wait, I need to double-check this reasoning.
    #
    # The claim is that if we partition the index sets, the rank of the full tensor is the
    # sum of the ranks of the sub-tensors. This is true when the factor matrices have a
    # block-diagonal structure corresponding to the partition.
    #
    # Let's verify with a simple case: 2x2x2 partitioned into four 1x1x1 blocks.
    # Rank should be 4*1 = 4. Schoolbook for 2x2x2 is 8. Strassen is 7.
    # But the partition approach gives 4, which is impossible since Strassen says 7 is minimal.
    # 
    # Ah, I see the issue. The partition approach only works when the sub-tensors can be
    # independently decomposed, but the factor matrices must be consistent. The rank of the
    # direct sum is indeed the sum of ranks, but the "sub-tensors" in the partition are NOT
    # the same as the multiplication tensors for the block sizes in the way I was thinking.
    #
    # Let me reconsider. The multiplication tensor T for C = AB has entries T[i,j,k] = 1
    # for all i,j,k. If we partition I, J, K, then T restricted to I_a x J_b x K_c is
    # all-ones, which is exactly the multiplication tensor for matrices of those sizes.
    # And the full tensor T is the direct sum (tensor product in the category of
    # trilinear forms) of these restricted tensors.
    #
    # Actually, I think the issue is that the "direct sum" here means that we can independently
    # choose rank-1 decompositions for each block, and the total rank is the sum. But the
    # factor matrices U, V, W would need to have a specific block structure.
    #
    # Let me just try to construct it and verify.
    
    def rank_nm_p(nn, mm, pp):
        """Get the rank for multiplying nn x mm by mm x pp."""
        if nn == 2 and mm == 2 and pp == 2:
            return 7
        return nn * mm * pp
    
    def get_decomp(nn, mm, pp):
        """Get U, V, W for nn x mm x pp multiplication."""
        if nn == 2 and mm == 2 and pp == 2:
            return _strassen_2x2()
        else:
            return _schoolbook(nn, mm, pp)
    
    def all_compositions(total, max_part=None):
        """Generate all compositions of total."""
        if max_part is None:
            max_part = total
        if total == 0:
            yield []
            return
        for first in range(1, max_part + 1):
            for rest in all_compositions(total - first, max_part):
                yield [first] + rest
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    deadline = time.time() + 120
    
    # Try partitions of I, J, K
    i_parts = list(all_compositions(n))
    j_parts = list(all_compositions(m))
    k_parts = list(all_compositions(p))
    
    for ip in i_parts:
        for jp in j_parts:
            for kp in k_parts:
                total_rank = 0
                for ib in ip:
                    for jb in jp:
                        for kb in kp:
                            total_rank += rank_nm_p(ib, jb, kb)
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    # Record this partition
                    best_partition = (ip, jp, kp)
    
    # Now construct the decomposition using the best partition
    if best_partition is not None:
        ip, jp, kp = best_partition
        
        # Build U, V, W by concatenating the block decompositions
        U_rows = []
        V_rows = []
        W_rows = []
        col_offset = 0
        
        # We need to carefully build the factor matrices
        # U has n*m rows, V has m*p rows, W has p*n rows
        # Each block (i_blk, j_blk, k_blk) contributes a sub-decomposition
        
        U_full = [[0]*best_rank for _ in range(n*m)]
        V_full = [[0]*best_rank for _ in range(m*p)]
        W_full = [[0]*best_rank for _ in range(p*n)]
        
        col = 0
        # Track offsets for each dimension
        i_offsets = []
        i_off = 0
        for ib in ip:
            i_offsets.append(i_off)
            i_off += ib
        
        j_offsets = []
        j_off = 0
        for jb in jp:
            j_offsets.append(j_off)
            j_off += jb
        
        k_offsets = []
        k_off = 0
        for kb in kp:
            k_offsets.append(k_off)
            k_off += kb
        
        for ia, ib_size in enumerate(ip):
            for ja, jb_size in enumerate(jp):
                for ka, kb_size in enumerate(kp):
                    U_blk, V_blk, W_blk = get_decomp(ib_size, jb_size, kb_size)
                    blk_rank = len(U_blk[0]) if U_blk else 0
                    
                    for r in range(blk_rank):
                        # U: row i*m+j where i is in block ia, j is in block ja
                        for ui in range(ib_size):
                            for uj in range(jb_size):
                                i_global = i_offsets[ia] + ui
                                j_global = j_offsets[ja] + uj
                                U_full[i_global * m + j_global][col + r] = U_blk[ui * jb_size + uj][r]
                        
                        # V: row j*p+k where j is in block ja, k is in block ka
                        for vj in range(jb_size):
                            for vk in range(kb_size):
                                j_global = j_offsets[ja] + vj
                                k_global = k_offsets[ka] + vk
                                V_full[j_global * p + k_global][col + r] = V_blk[vj * kb_size + vk][r]
                        
                        # W: row k*n+i where k is in block ka, i is in block ia
                        for wk in range(kb_size):
                            for wi in range(ib_size):
                                k_global = k_offsets[ka] + wk
                                i_global = i_offsets[ia] + wi
                                W_full[k_global * n + i_global][col + r] = W_blk[wk * ib_size + wi][r]
                    
                    col += blk_rank
        
        # Verify
        if col == best_rank and _verify(U_full, V_full, W_full, n, m, p):
            U_full = _clean_matrix(U_full)
            V_full = _clean_matrix(V_full)
            W_full = _clean_matrix(W_full)
            return {"U": U_full, "V": V_full, "W": W_full}
        else:
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    else:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

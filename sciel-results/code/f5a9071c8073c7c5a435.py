import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly for integer/half-integer entries."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    if np.max(np.abs(T_computed - target)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _winograd_5x5():
    """
    Winograd frequency-domain decomposition for 5x5 matrix multiplication.
    
    The 5-point DFT has entries in Z[ζ_5] where ζ_5 = e^(2πi/5).
    The minimal polynomial of ζ_5 is x^4 + x^3 + x^2 + x + 1 = 0.
    So ζ_5^4 = -ζ_5^3 - ζ_5^2 - ζ_5 - 1.
    
    For the Winograd method, we use the fact that matrix multiplication
    C = A * B can be written as:
      C = F^{-1} * (F(A) ∘ F(B)) * F^{-T}
    where F is the 5-point DFT matrix.
    
    The key insight: if we can represent F and F^{-1} with rational (half-integer)
    entries, we get an exact decomposition.
    
    For n=5 (prime), the DFT matrix F has entries F[k,j] = ζ_5^(kj).
    F * F^H = 5 * I, so F^{-1} = (1/5) * F^H.
    
    The tensor decomposition:
      T_{a,b,c} = (1/25) * sum_k F[a,k] * F[b,k] * F[c,k]  ... this isn't quite right.
    
    Actually, let me think more carefully.
    
    Matrix mult: C[a,c] = sum_b A[a,b] * B[b,c]
    
    Winograd: 
      A' = F * A * F^H / 5  (but this is for the full transform)
    
    Actually the standard Winograd approach for matmul:
      Let F be the 5x5 DFT matrix.
      C = A*B
      
      In frequency domain:
      Let Â = F * A (row transform)
      Let B̂ = B * F^H (column transform)  
      Then Ĉ = Â * B̂ (element-wise? No, that's not right either)
    
    Let me reconsider. The tensor for matrix multiplication is:
      T_{(a,b),(b,c),(c,a)} = 1
      
    The rank-r decomposition is:
      T_{i,j,k} = sum_r U[i,r] * V[j,r] * W[k,r]
    where i = (a,b), j = (b,c), k = (c,a).
    
    For the Winograd/Fourier approach:
      We want U[(a,b), k] * V[(b,c), k] * W[(c,a), k] summed over k.
      
      If we set:
        U[(a,b), k] = F[a, k] * F[b, k]  (or some variant)
        V[(b,c), k] = ...
        W[(c,a), k] = ...
      
      Then sum_k U[(a,b),k] * V[(b,c),k] * W[(c,a),k]
      = sum_k F[a,k] * F[b,k] * V[(b,c),k] * W[(c,a),k]
      
    This is getting complex. Let me use a different, well-known approach.
    
    Actually, for the 5x5 case, the Winograd algorithm gives rank 25:
      C = A * B where A, B, C are 5x5.
      
      The idea: use the 5-point convolution via FFT.
      A 5x5 matmul can be viewed as 5 row-vector × 5-column-vector operations.
      Each such operation (length-5 dot product) can be done with 5 multiplications
      in the frequency domain (since 5 is prime, the DFT diagonalizes circular convolution).
      
      But actually, a length-n dot product has rank n (you can't do better for exact computation
      with the standard approach). So 5 rows * 5 columns = 25 scalar multiplications.
      
      Wait, that's just schoolbook in a different form. The rank is still 125 for the full 5x5x5.
      
      Hmm, let me reconsider. The Winograd method for n×n matrix multiplication:
      For n=5, the standard Winograd algorithm uses the fact that:
        C = A * B = F^{-1} * (F * A) ∘ (B * F^{-T}) * F
      where ∘ is element-wise product... no, that's not standard.
      
    Let me look at this differently. The tensor rank of 5×5×5 matmul:
      - Schoolbook: rank 125
      - The known lower bound is at least 5^3 = 125? No, the lower bound is 2*5^3 - 1 = 124? 
        Actually, the bilinear complexity of n×n matmul is at most n^3 (schoolbook) and at least 
        something related to the rank of the tensor.
        
      For n=2: rank 7 (Strassen)
      For n=3: rank 23 (Laderman) 
      For n=4: rank 49 (or better?)
      For n=5: unknown exact minimum, but upper bounds exist.
      
    The Winograd method for 5×5: 
      Actually, I recall that for prime p, the p×p matrix multiplication can be done
      with rank p^2 using the Winograd/Fourier method when we allow the factors to have
      entries in the field containing the p-th roots of unity. But the problem requires
      integer or half-integer entries.
      
    Key insight: The 5-point DFT matrix has entries ζ_5^(kj) where ζ_5 = e^(2πi/5).
    These are NOT rational numbers. So we cannot directly use the DFT matrix with
    rational entries.
    
    However, we can use the fact that ζ_5 satisfies ζ_5^4 + ζ_5^3 + ζ_5^2 + ζ_5 + 1 = 0.
    We can represent each ζ_5^k as a vector in the basis {1, ζ_5, ζ_5^2, ζ_5^3} over Q.
    
    But this would make the factor matrices have entries that are vectors, not scalars.
    
    Alternative: Use the "Winograd" approach where we split the 5×5 matmul into
    overlapping 4×4 blocks or use the specific structure.
    
    Actually, let me try a completely different approach. Let me use the known
    Laderman 3×3 decomposition combined with block methods.
    
    Laderman 3×3: rank 23
    Block 5×5 as [3,2] × [3,2] × [3,2]:
      - 3×3×3 blocks: 8 blocks, each rank 23 → 8*23 = 184 (worse than schoolbook)
      - But cross-blocks can use Strassen or other methods.
    
    Hmm, this is getting complicated. Let me try a more direct approach.
    
    Actually, for the Winograd method to work with rational entries, we need to use
    the fact that the 5-point DFT can be represented using the "fast Winograd"
    algorithm which uses only additions and a small number of multiplications.
    
    For a length-5 convolution, the Winograd algorithm uses:
      5 multiplications (since 5 is prime, we need 5)
    
    So for 5×5 matmul: 5 rows × 5 cols × 5 mults each = 125. Same as schoolbook.
    
    The Winograd method doesn't help for prime lengths in terms of reducing the
    number of multiplications below n for an n-point convolution.
    
    So the Winograd/Fourier approach does NOT give a rank-25 decomposition for 5×5×5
    with rational entries. My initial assumption was wrong.
    
    Let me instead focus on what CAN work:
    1. Use Laderman 3×3 (rank 23) in a block decomposition
    2. Use Strassen 2×2 (rank 7) in a block decomposition
    3. Try to find a better 5×5 decomposition
    
    For 5×5 with blocks [3,2]×[3,2]×[3,2]:
      Diagonal blocks: 3×3×3 (rank 23 each, 8 of them) + 2×2×2 (rank 7 each, 8 of them)
      But wait, it's not that simple. The block structure is:
        A = [A11 A12; A21 A22] where A11 is 3×3, A12 is 3×2, etc.
        B = [B11 B12; B21 B22]
        C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22;
                   A21*B11 + A22*B21, A21*B12 + A22*B22]
      
      So we need:
        A11*B11 (3×3 * 3×3): rank 23
        A12*B21 (3×2 * 2×3): rank 18 (schoolbook) or can we do better?
        A11*B12 (3×3 * 3×2): rank 18
        A12*B22 (3×2 * 2×2): rank 12
        A21*B11 (2×3 * 3×3): rank 18
        A22*B21 (2×2 * 2×3): rank 12
        A21*B12 (2×3 * 3×2): rank 18
        A22*B22 (2×2 * 2×2): rank 7
      
      Total: 23 + 18 + 18 + 12 + 18 + 12 + 18 + 7 = 126. Worse than schoolbook!
      
    The issue is that the rectangular blocks don't benefit from Strassen/Laderman.
    
    Let me try [2,2,1]×[2,2,1]×[2,2,1]:
      27 blocks, each 2×2×2 (rank 7) or 2×2×1, 2×1×2, 1×2×2, etc.
      
      This is what the current program tries. Let me check what rank it achieves.
      
    Actually, let me just implement the best known approach: use Strassen for all
    2×2×2 sub-blocks and schoolbook for the rest, with optimal partitioning.
    
    For 5×5×5 with partition [2,2,1]:
      Number of 2×2×2 blocks: 3*3*3 = 27, but only those where all three dimensions
      are from the "2" parts.
      
      Let me count:
      Partitions of 5 into [2,2,1]:
      - 2×2×2: positions (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        That's 8 blocks of size 2×2×2, each rank 7 → 56
      - 2×2×1: 8 blocks, each rank 4 → 32
      - 2×1×2: 8 blocks, each rank 4 → 32  
      - 2×1×1: 8 blocks, each rank 2 → 16
      - 1×2×2: 8 blocks, each rank 4 → 32
      - 1×2×1: 8 blocks, each rank 2 → 16
      - 1×1×2: 8 blocks, each rank 2 → 16
      - 1×1×1: 8 blocks, each rank 1 → 8
      
      Wait, I'm double-counting. Let me be more careful.
      
      With partition [2,2,1] for each of A, B, C:
      Block (i,j,k) has size sizes_a[i] × sizes_b[j] × sizes_c[k]
      
      i,j,k each in {0,1,2} where sizes = [2,2,1]
      
      (0,0,0): 2×2×2, rank 7
      (0,0,1): 2×2×1, rank 4
      (0,1,0): 2×2×2, rank 7
      (0,1,1): 2×2×1, rank 4
      (0,2,0): 2×1×2, rank 4
      (0,2,1): 2×1×1, rank 2
      (1,0,0): 2×2×2, rank 7
      (1,0,1): 2×2×1, rank 4
      (1,1,0): 2×2×2, rank 7
      (1,1,1): 2×2×1, rank 4
      (1,2,0): 2×1×2, rank 4
      (1,2,1): 2×1×1, rank 2
      (2,0,0): 1×2×2, rank 4
      (2,0,1): 1×2×1, rank 2
      (2,1,0): 1×2×2, rank 4
      (2,1,1): 1×2×1, rank 2
      (2,2,0): 1×1×2, rank 2
      (2,2,1): 1×1×1, rank 1
      
      Total: 7*4 + 4*8 + 2*4 + 1*1 = 28 + 32 + 8 + 1 = 69
      
      Wait let me recount:
      2×2×2: (0,0,0), (0,1,0), (1,0,0), (1,1,0) → 4 blocks, rank 7 each → 28
      2×2×1: (0,0,1), (0,1,1), (1,0,1), (1,1,1) → 4 blocks, rank 4 each → 16
      2×1×2: (0,2,0), (1,2,0) → 2 blocks, rank 4 each → 8
      2×1×1: (0,2,1), (1,2,1) → 2 blocks, rank 2 each → 4
      1×2×2: (2,0,0), (2,1,0) → 2 blocks, rank 4 each → 8
      1×2×1: (2,0,1), (2,1,1) → 2 blocks, rank 2 each → 4
      1×1×2: (2,2,0) → 1 block, rank 2 → 2
      1×1×1: (2,2,1) → 1 block, rank 1 → 1
      
      Total: 28 + 16 + 8 + 4 + 8 + 4 + 2 + 1 = 71
      
      Hmm, that's better than 125. Let me verify with the code.
    
    Actually, the current program already tries this and gets rank 116 according to the
    knowledge. So something is off in my counting, or the implementation is different.
    
    Let me just implement a clean version that tries multiple partitions and uses
    Strassen for 2×2×2 and schoolbook otherwise, and returns the best result.
    """
    n = 5
    # DFT matrix for 5 points - but we can't use irrational entries
    # So Winograd with rational entries doesn't directly work for rank 25.
    
    # Instead, let's use the best block decomposition available.
    # Return schoolbook as fallback since Winograd doesn't give rational rank-25.
    U = np.zeros((25, 125), dtype=np.int64)
    V = np.zeros((25, 125), dtype=np.int64)
    W = np.zeros((25, 125), dtype=np.int64)
    col = 0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                U[a*5 + b, col] = 1
                V[b*5 + c, col] = 1
                W[c*5 + a, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _laderman_3x3():
    """Laderman 3x3x3 decomposition, rank 23."""
    # Laderman's algorithm for 3x3 matrix multiplication
    # Reference: Laderman, "A nonclassical 3x3 matrix multiplication"
    U = np.zeros((9, 23), dtype=np.int64)
    V = np.zeros((9, 23), dtype=np.int64)
    W = np.zeros((9, 23), dtype=np.int64)
    
    # M1 = A11*(B11+B22-B33)
    U[0,0]=1; V[0,0]=1; V[4,0]=1; V[8,0]=-1; W[0,0]=1
    # M2 = (A11+A12)*(B22-B33)
    U[0,1]=1; U[1,1]=1; V[4,1]=1; V[8,1]=-1; W[1,1]=1
    # M3 = A22*(B11-B33+B22)
    U[4,2]=1; V[0,2]=1; V[8,2]=-1; V[4,2]=1; W[4,2]=1
    # M4 = (A22+A23)*(B11-B33)
    U[4,3]=1; U[5,3]=1; V[0,3]=1; V[8,3]=-1; W[5,3]=1
    # M5 = A33*(B11+B22-B33)
    U[8,4]=1; V[0,4]=1; V[4,4]=1; V[8,4]=-1; W[8,4]=1
    # M6 = (A33+A31)*(B22-B33)
    U[8,5]=1; U[3,5]=1; V[4,5]=1; V[8,5]=-1; W[3,5]=1
    # M7 = A11*(B22+B33)
    U[0,6]=1; V[4,6]=1; V[8,6]=1; W[0,6]=1
    # M8 = (A11+A12)*(B33+B22)
    U[0,7]=1; U[1,7]=1; V[8,7]=1; V[4,7]=1; W[1,7]=1
    # M9 = A22*(B33+B11)
    U[4,8]=1; V[8,8]=1; V[0,8]=1; W[4,8]=1
    # M10 = (A22+A23)*(B11+B33)
    U[4,9]=1; U[5,9]=1; V[0,9]=1; V[8,9]=1; W[5,9]=1
    # M11 = A33*(B11+B22)
    U[8,10]=1; V[0,10]=1; V[4,10]=1; W[8,10]=1
    # M12 = (A33+A31)*(B22+B11)
    U[8,11]=1; U[3,11]=1; V[4,11]=1; V[0,11]=1; W[3,11]=1
    # M13 = A12*B11
    U[1,12]=1; V[0,12]=1; W[1,12]=1
    # M14 = A23*B22
    U[5,13]=1; V[4,13]=1; W[5,13]=1
    # M15 = A31*B33
    U[3,14]=1; V[8,14]=1; W[3,14]=1
    # M16 = A13*B11
    U[2,15]=1; V[0,15]=1; W[2,15]=1
    # M17 = A21*B22
    U[4,16]=1; V[4,16]=1; W[4,16]=1
    # M18 = A32*B33
    U[7,17]=1; V[8,17]=1; W[7,17]=1
    # M19 = A12*B33
    U[1,18]=1; V[8,18]=1; W[1,18]=1
    # M20 = A23*B11
    U[5,19]=1; V[0,19]=1; W[5,19]=1
    # M21 = A31*B22
    U[3,20]=1; V[4,20]=1; W[3,20]=1
    # M22 = A13*B33
    U[2,21]=1; V[8,21]=1; W[2,21]=1
    # M23 = A21*B11
    U[4,22]=1; V[0,22]=1; W[4,22]=1
    
    return U, V, W

def _block_decompose(n, m, p, sizes_a, sizes_b, sizes_c):
    """
    Block decomposition for n×m×p matrix multiplication.
    Uses Strassen for 2×2×2, Laderman for 3×3×3, schoolbook otherwise.
    """
    U_terms = []
    V_terms = []
    W_terms = []
    rank_offset = 0

    for i in range(len(sizes_a)):
        for j in range(len(sizes_b)):
            for k in range(len(sizes_c)):
                sa, sb, sc = sizes_a[i], sizes_b[j], sizes_c[k]
                if sa == 0 or sb == 0 or sc == 0:
                    continue

                if sa == 2 and sb == 2 and sc == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                    r = 7
                elif sa == 3 and sb == 3 and sc == 3:
                    U_blk, V_blk, W_blk = _laderman_3x3()
                    r = 23
                else:
                    r = sa * sb * sc
                    U_blk = np.zeros((sa*sb, r), dtype=np.int64)
                    V_blk = np.zeros((sb*sc, r), dtype=np.int64)
                    W_blk = np.zeros((sc*sa, r), dtype=np.int64)
                    col = 0
                    for a in range(sa):
                        for b in range(sb):
                            for c in range(sc):
                                U_blk[a*sb+b, col] = 1
                                V_blk[b*sc+c, col] = 1
                                W_blk[c*sa+a, col] = 1
                                col += 1

                row_start_A = sum(sizes_a[:i])
                row_start_B = sum(sizes_b[:j])
                row_start_C = sum(sizes_c[:k])

                for rr in range(r):
                    for a_idx in range(sa*sb):
                        val = U_blk[a_idx, rr]
                        if val != 0:
                            U_terms.append((row_start_A + a_idx, rank_offset + rr, val))
                    for b_idx in range(sb*sc):
                        val = V_blk[b_idx, rr]
                        if val != 0:
                            V_terms.append((row_start_B + b_idx, rank_offset + rr, val))
                    for c_idx in range(sc*sa):
                        val = W_blk[c_idx, rr]
                        if val != 0:
                            W_terms.append((row_start_C + c_idx, rank_offset + rr, val))
                rank_offset += r

    total_rank = rank_offset
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    for row, col, val in U_terms:
        U[row, col] += val
    for row, col, val in V_terms:
        V[row, col] += val
    for row, col, val in W_terms:
        W[row, col] += val
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try various block decompositions
    partitions = [
        ([2, 2, 1], [2, 2, 1], [2, 2, 1]),
        ([2, 1, 2], [2, 1, 2], [2, 1, 2]),
        ([1, 2, 2], [1, 2, 2], [1, 2, 2]),
        ([2, 2, 1], [2, 2, 1], [1, 2, 2]),
        ([2, 2, 1], [1, 2, 2], [2, 2, 1]),
        ([1, 2, 2], [2, 2, 1], [2, 2, 1]),
        ([3, 2], [3, 2], [3, 2]),
        ([2, 3], [2, 3], [2, 3]),
        ([2, 2, 1], [3, 2], [2, 3]),
        ([3, 2], [2, 2, 1], [2, 3]),
        ([2, 3], [2, 2, 1], [3, 2]),
    ]
    
    for sa, sb, sc in partitions:
        if time.time() > deadline - 30:
            break
        U_blk, V_blk, W_blk, blk_rank = _block_decompose(n, m, p, sa, sb, sc)
        if blk_rank < best_rank:
            if _verify(U_blk, V_blk, W_blk, n, m, p, blk_rank):
                best_rank = blk_rank
                best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
                print(f"Block decomposition {sa}|{sb}|{sc} rank: {best_rank}")
            else:
                print(f"Block decomposition {sa}|{sb}|{sc} rank {blk_rank} FAILED")
    
    return best_result
# EVOLVE-BLOCK-END

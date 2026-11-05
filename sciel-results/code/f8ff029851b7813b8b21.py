import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices into S1={0,1}, S2={2,3}, S3={4}
    # Sizes: 2, 2, 1
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # We will collect terms as (i, j, k, coeff) or directly build U, V, W
    # Since rank is small, we can build the matrices directly.
    # Max rank estimate: 73.
    # Let's use a dictionary to merge identical terms if any, though unlikely with distinct blocks.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(i, j, k, coeff=1):
        # i, j, k are global indices 0..4
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U_terms.append((i * m + j, coeff))
        V_terms.append((j * p + k, coeff))
        W_terms.append((k * n + i, coeff))

    # Helper to add Strassen terms for 2x2x2
    # Local indices 0,1 for rows/cols of the 2x2 blocks
    # Maps local (li, lj, lk) to global (gi, gj, gk)
    def add_strassen(gi_base, gj_base, gk_base):
        # gi_base, gj_base, gk_base are the starting indices for the 2x2 blocks
        # The blocks are A[gi:gi+2, gj:gj+2], B[gj:gj+2, gk:gk+2], C[gk:gk+2, gi:gi+2]
        # Strassen 7 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add terms for the tensor decomposition.
        # Each Strassen term M_r is a product of a linear combo of A and a linear combo of B.
        # This results in multiple (i,j,k) terms for the global tensor.
        
        # M1: (A00+A11)(B00+B11)
        # A00 is at (gi+0, gj+0), A11 at (gi+1, gj+1)
        # B00 is at (gj+0, gk+0), B11 at (gj+1, gk+1)
        # Products:
        # A00*B00 -> (gi+0, gj+0, gk+0) coeff 1
        # A00*B11 -> (gi+0, gj+1, gk+1) coeff 1
        # A11*B00 -> (gi+1, gj+0, gk+0) coeff 1
        # A11*B11 -> (gi+1, gj+1, gk+1) coeff 1
        
        # We add these to the global terms.
        # Note: The Strassen algorithm computes C using these Ms.
        # The tensor decomposition must replicate the exact multiplication.
        # The standard Strassen decomposition of the 2x2x2 tensor has rank 7.
        # The 7 terms are the rank-1 decompositions of the tensor.
        
        # I will use the explicit 7 rank-1 tensors for 2x2x2.
        # Reference: Strassen's 7-term algorithm.
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # This is not a single rank-1 tensor.
        # The rank-1 tensors are the "basis" for the decomposition.
        
        # Let's use the known 7 vectors for U, V, W for 2x2x2.
        # U (2x2 x 7), V (2x2 x 7), W (2x2 x 7)
        
        # U:
        # T1: A00+A11 -> [[1,0],[0,1]]
        # T2: A10+A11 -> [[0,0],[1,1]]
        # T3: A00 -> [[1,0],[0,0]]
        # T4: A11 -> [[0,0],[0,1]]
        # T5: A00+A01 -> [[1,1],[0,0]]
        # T6: A10-A00 -> [[-1,0],[1,0]]
        # T7: A01-A11 -> [[0,1],[0,-1]]
        
        # V:
        # T1: B00+B11 -> [[1,0],[0,1]]
        # T2: B00 -> [[1,0],[0,0]]
        # T3: B10-B11 -> [[0,-1],[1,0]]
        # T4: B01-B10 -> [[0,1],[-1,0]]
        # T5: B11 -> [[0,0],[0,1]]
        # T6: B00+B10 -> [[1,0],[1,0]]
        # T7: B01+B11 -> [[0,1],[0,1]]
        
        # W:
        # T1: C00+C11? No, W corresponds to the output C.
        # The decomposition is T = sum u_r v_r w_r^T?
        # W is the output factor.
        # T1: C00 + C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So W1 contributes to C00 and C11 with coeff 1.
        # W2 contributes to C10 with 1, C11 with -1.
        # W3 contributes to C01 with 1, C11 with 1.
        # W4 contributes to C00 with 1, C10 with 1.
        # W5 contributes to C00 with -1, C01 with 1.
        # W6 contributes to C11 with 1.
        # W7 contributes to C00 with 1.
        
        # Let's verify the sum for C00:
        # M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? No.
        # C00 = M1 + M4 - M5 + M7.
        # Coeffs: T1:1, T4:1, T5:-1, T7:1.
        
        # We will add these 7 terms for each 2x2x2 block.
        
        u_mat = [
            [[1,0],[0,1]], # T1
            [[0,0],[1,1]], # T2
            [[1,0],[0,0]], # T3
            [[0,0],[0,1]], # T4
            [[1,1],[0,0]], # T5
            [[-1,0],[1,0]],# T6
            [[0,1],[0,-1]] # T7
        ]
        v_mat = [
            [[1,0],[0,1]], # T1
            [[1,0],[0,0]], # T2
            [[0,-1],[1,0]],# T3
            [[0,1],[-1,0]],# T4
            [[0,0],[0,1]], # T5
            [[1,0],[1,0]], # T6
            [[0,1],[0,1]]  # T7
        ]
        w_mat = [
            [[1,0],[0,1]], # T1: C00, C11
            [[0,0],[1,-1]],# T2: C10, C11
            [[0,1],[1,0]], # T3: C01, C11? No, C01=1, C11=1.
                           # Wait, C01 = M3+M5. C11 = M1-M2+M3+M6.
                           # T3 coeff in C01 is 1. In C11 is 1.
                           # So W3 = [[0,1],[1,0]]? No, W is 2x2.
                           # W3[0,0]=0, W3[0,1]=1 (C01), W3[1,0]=1 (C11? No, C11 is [1,1]).
                           # Let's map C indices to W matrix indices.
                           # C is 2x2. C[0,0], C[0,1], C[1,0], C[1,1].
                           # W[r] is a 2x2 matrix.
                           # T3: C01 += 1, C11 += 1.
                           # W3 = [[0,1],[0,1]]? No, C11 is at [1,1].
                           # W3[0,1] = 1. W3[1,1] = 1.
                           # So W3 = [[0,1],[0,1]].
            [[1,0],[1,0]], # T4: C00=1, C10=1. W4[0,0]=1, W4[1,0]=1.
            [[-1,1],[0,0]],# T5: C00=-1, C01=1. W5[0,0]=-1, W5[0,1]=1.
            [[0,0],[0,1]], # T6: C11=1. W6[1,1]=1.
            [[1,0],[0,0]]  # T7: C00=1. W7[0,0]=1.
        ]
        
        # Let's re-verify W coefficients.
        # C00: T1(1), T4(1), T5(-1), T7(1). Sum 2? No.
        # C00 = M1 + M4 - M5 + M7.
        # T1: 1. T4: 1. T5: -1. T7: 1.
        # Sum: 1+1-1+1 = 2. This is wrong. C00 should be 1.
        # Ah, the Strassen formula is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check T1 (M1).
        # M1 contributes to C00 (+1) and C11 (+1).
        # So W1[0,0]=1, W1[1,1]=1.
        
        # T2 (M2).
        # M2 contributes to C10 (+1) and C11 (-1).
        # W2[1,0]=1, W2[1,1]=-1.
        
        # T3 (M3).
        # M3 contributes to C01 (+1) and C11 (+1).
        # W3[0,1]=1, W3[1,1]=1.
        
        # T4 (M4).
        # M4 contributes to C00 (+1) and C10 (+1).
        # W4[0,0]=1, W4[1,0]=1.
        
        # T5 (M5).
        # M5 contributes to C00 (-1) and C01 (+1).
        # W5[0,0]=-1, W5[0,1]=1.
        
        # T6 (M6).
        # M6 contributes to C11 (+1).
        # W6[1,1]=1.
        
        # T7 (M7).
        # M7 contributes to C00 (+1).
        # W7[0,0]=1.
        
        # Let's check C00 sum: 1 (T1) + 1 (T4) - 1 (T5) + 1 (T7) = 2.
        # This implies the standard Strassen formula I recalled might have a factor of 1/2 or I am missing a sign.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My W coefficients match this.
        # Why sum 2?
        # Because the tensor decomposition sums the products.
        # The product $U_r V_r W_r$ is added.
        # For C00, the coefficient is $\sum_r U_{00,r} V_{00,r} W_{00,r}$.
        # This is not just the sum of W coefficients.
        # It is the sum of $U_{00,r} V_{00,r} W_{00,r}$.
        
        # So I just need to add the terms $U_{li, lj, r} V_{lj, lk, r} W_{lk, li, r}$?
        # No, the tensor entry is $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        
        # I will iterate r=0..6, and for each r, iterate local indices li, lj, lk.
        # Add $U_{li, lj, r} V_{lj, lk, r} W_{lk, li, r}$ to the global term at (gi+li, gj+lj, gk+lk).
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        coeff = u_mat[r][li][lj] * v_mat[r][lj][lk] * w_mat[r][lk][li]
                        if coeff != 0:
                            gi = gi_base + li
                            gj = gj_base + lj
                            gk = gk_base + lk
                            add_term(gi, gj, gk, coeff)

    # Helper to add rank-1 terms for m x n x p
    # Here we assume the tensor is all 1s (standard matmul)
    # Rank is 1 for m x n x p if we can factor it?
    # No, rank of m x n x p matmul is mn if p=1?
    # For 2x1x2, rank 1.
    # For 2x2x1, rank 1?
    # Let's assume we can use rank 1 for these small blocks.
    # If not, we can use schoolbook.
    
    def add_rank1(gi_base, gj_base, gk_base, ni, nj, nk):
        # ni, nj, nk are sizes
        # If rank is 1, we just add one term with coeff 1?
        # No, the tensor is $T_{i,j,k} = 1$.
        # If rank 1, $T = u \otimes v \otimes w$.
        # $u_{i} v_{j} w_{k} = 1$.
        # This implies $u, v, w$ are all 1s.
        # So we add one term: (gi_base, gj_base, gk_base) with coeff 1?
        # No, we need to cover all i,j,k.
        # If $u_i=1, v_j=1, w_k=1$, then $T_{i,j,k}=1$ for all i,j,k.
        # So we add ONE term for the whole block?
        # Yes, if the tensor is all 1s.
        
        # Wait, the indices are local.
        # We need to add the term for the specific (i,j,k) in the global space.
        # But the term is a single rank-1 component.
        # It contributes to ALL (i,j,k) in the block.
        # So we add ONE entry to U, V, W.
        # But U, V, W are vectors for the whole matrix.
        # So we need to set $U_{i,j,r} = 1$ for all i,j in the block?
        # No, $U$ is a matrix of size $25 \times R$.
        # For a rank-1 term, $U$ is a vector, $V$ is a vector, $W$ is a vector.
        # $U_{i,j} = u_i v_j$? No.
        # The decomposition is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        # If we use one term $r$, $U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ for all i,j,k in the block.
        # This requires $U_{i,j,r} = a_{i,j}$, $V_{j,k,r} = b_{j,k}$, $W_{k,i,r} = c_{k,i}$ such that $a_{i,j} b_{j,k} c_{k,i} = 1$.
        # This is possible if we set $a_{i,j}=1, b_{j,k}=1, c_{k,i}=1$.
        # So we add ONE term.
        # We set $U_{i,j,r}=1$ for all i,j in the block?
        # No, $U$ has one column for this term.
        # So we set $U_{i,j, r} = 1$ for all i in block_i, j in block_j.
        # Similarly for V and W.
        
        # But wait, if I add one term, it contributes 1 to ALL (i,j,k) in the block.
        # This is correct for the standard matmul tensor.
        
        # So I just need to add one term, and mark the relevant entries in U, V, W.
        # However, my `add_term` function adds a single (i,j,k) point.
        # I need a function to add a "block term".
        
        pass

    # This approach is getting complicated to manage the rank-1 blocks vs Strassen blocks.
    # I will stick to the 73-rank decomposition using Strassen for 2x2x2 and schoolbook for others.
    # Schoolbook for 2x1x2 has rank 2.
    # Schoolbook for 2x2x1 has rank 2.
    
    # Let's re-calculate rank with schoolbook for non-2x2x2.
    # C11 (2x2): 7 (Strassen) + 7 (Strassen) + 2 (Schoolbook 2x1x2) = 16?
    # Wait, A13 is 2x1, B31 is 1x2.
    # $C_{11}$ block is 2x2.
    # Term 3: $A_{13} B_{31}$.
    # $A_{13}$ is 2x1. $B_{31}$ is 1x2.
    # $C_{ik} = \sum_{j=0}^0 A_{ij} B_{jk}$.
    # $j$ is only 0.
    # $C_{ik} = A_{i0} B_{0k}$.
    # This is a rank 1 tensor?
    # $T_{i,0,k} = 1$.
    # Rank 1.
    # So rank is 1.
    
    # So C11 rank = 7 + 7 + 1 = 15.
    # C12 rank = 7 + 7 + 1 = 15.
    # C13 rank = 1 (2x2x1) + 1 (2x2x1) + 1 (2x1x1) = 3.
    # C21 rank = 15.
    # C22 rank = 15.
    # C23 rank = 3.
    # C31 rank = 3.
    # C32 rank = 3.
    # C33 rank = 1.
    
    # Total 73.
    
    # I will implement this.
    
    return {"U": None, "V": None, "W": None} # Placeholder
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a blocking strategy to reduce rank.
    # Split m=4 into two blocks of size 2.
    # For each block, split n=3 into 1 and 2.
    # 
    # Block 1 (m=2):
    #   Sub-block 1 (n=1, m=2, p=7): Rank 2
    #   Sub-block 2 (n=2, m=2, p=7): Rank 22 (3 * Strassen(2x2x2) + 1 * (2x2x1))
    #   Total for n=3, m=2, p=7 is 24.
    #
    # Total Rank = 24 + 24 = 48.
    
    # We construct the tensors explicitly.
    # T = sum_c U_c \otimes V_c \otimes W_c
    # U is (n*m) x Rank
    # V is (m*p) x Rank
    # W is (p*n) x Rank
    
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # We need to decompose the multiplication.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We iterate over the two m-blocks (j=0,1 and j=2,3)
    for block in range(2):
        # j_offset is 0 or 2
        j_off = block * 2
        
        # Sub-block 1: n=1 (i=0), m=2 (j=j_off, j_off+1), p=7
        # Rank 2 decomposition for 1x2x7
        # T[i=0, j, k] = A[0,j] B[j,k]
        # Term 1: j=j_off
        # Term 2: j=j_off+1
        # Actually, for 1x2x7, we can just use 2 terms?
        # T[0, j_off, k] = A[0, j_off] B[j_off, k]
        # T[0, j_off+1, k] = A[0, j_off+1] B[j_off+1, k]
        # These are two separate rank-1 tensors?
        # Tensor 1: u[0]=1, v[j_off]=1, w[k]=B[j_off,k]? No, w must be constant for the term.
        # The term is $A[0, j] B[j, k]$.
        # This is not a single rank-1 tensor in the decomposition sense unless we fix k.
        # Wait, the decomposition is $T_{ijk} = \sum_c U_{ic} V_{jc} W_{kc}$.
        # If we set $c=1$, $U_{01}=A_{0j}, V_{j1}=1, W_{k1}=B_{jk}$?
        # No, $U, V, W$ must be constant matrices, not depending on A and B.
        # The decomposition is for the **tensor of the operation**, not for specific matrices.
        # The tensor $T$ has entries $T_{ijk} = 1$ if the term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
        # Actually, $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
        # So $T_{ijk} = 1$ for all valid i,j,k.
        # We need to decompose the tensor $T$ where $T_{ijk} = 1$ if i,j,k are valid indices.
        # Wait, the tensor for matrix multiplication is $T_{ijk} = \delta_{ik} \dots$?
        # No. $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
        # We want to write $T = \sum_{c=1}^R u_c \otimes v_c \otimes w_c$.
        # This means $1 = \sum_{c} U_{i c} V_{j c} W_{k c}$ for all i,j,k?
        # No, that's not right.
        # The condition is:
        # $\sum_{c} U_{i c} V_{j c} W_{k c} = \delta_{?}$
        # Actually, the standard form is:
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # This is equivalent to $\sum_{j} A_{ij} B_{jk} = \sum_{c} (\sum_{i'} U_{i' c} A_{i' ?}) \dots$
        # Let's stick to the interface:
        # U, V, W are fixed matrices.
        # The condition is that the tensor defined by $T_{ijk} = \sum_c U_{i c} V_{j c} W_{k c}$
        # equals the matrix multiplication tensor.
        # The matrix multiplication tensor $T^{mm}_{ijk}$ is 1 if the indices correspond to a valid product term?
        # No, the tensor $T^{mm}$ is such that $\langle T^{mm}, A \otimes B \rangle = \text{trace}(C \dots)$?
        # The problem states: "exactly decompose the matrix multiplication tensor".
        # The tensor is $T_{ijk} = 1$ if we consider the basis?
        # Actually, $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in the expression for $C_{ik}$.
        # So $T_{ijk} = 1$ for all i,j,k.
        # Wait, if $T_{ijk}=1$, then $\sum_c U_{ic} V_{jc} W_{kc} = 1$ for all i,j,k.
        # This implies $U V^T W^T$? No.
        # It implies that the tensor is the all-ones tensor.
        # But matrix multiplication is not all-ones.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # This is a bilinear map.
        # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # No, the tensor $T$ satisfies $T(A,B) = C$.
        # $T_{ijk}$ is 1 if the term $A_{ij} B_{jk}$ is used.
        # So yes, $T_{ijk} = 1$ for all valid i,j,k.
        # So we need $\sum_c U_{ic} V_{jc} W_{kc} = 1$ for all i,j,k.
        
        # This is a very strong condition.
        # If $U, V, W$ are 0/1 matrices, this means the product of the matrices must be 1 everywhere?
        # No, the sum is over c.
        # This is equivalent to $U V^T W^T$? No.
        # Let $M_{ij} = \sum_c U_{ic} V_{jc}$.
        # Then $\sum_k M_{ij} W_{kc} = 1$? No, the sum is over c, and k is an index of W.
        # $\sum_c U_{ic} V_{jc} W_{kc} = 1$.
        # This means that for each k, the matrix $W_k^T U V^T$?
        # Actually, if we fix k, $\sum_c (U_{ic} V_{jc}) W_{kc} = 1$.
        # Let $Y_{ij} = \sum_c U_{ic} V_{jc}$.
        # Then $\sum_c Y_{ij} W_{kc} = 1$? No, $Y_{ij}$ depends on c.
        # This structure is hard to satisfy with 1s.
        
        # However, the schoolbook solution uses $U, V, W$ with 1s.
        # For schoolbook, $c$ corresponds to $(i,j,k)$.
        # $U_{i'm,c} = 1$ if $i'=i, m'=j$.
        # $V_{j'm,c} = 1$ if $j'=j, m'=k$.
        # $W_{k'n,c} = 1$ if $k'=k, n'=i$.
        # Then $\sum_c U_{i c} V_{j c} W_{k c}$ is 1 only if $c$ matches $(i,j,k)$, else 0.
        # So $\sum_c \dots = 1$ if we sum over the specific c?
        # No, the condition is that the **decomposition** reconstructs the tensor.
        # The tensor is $T_{ijk} = 1$.
        # So we need $\sum_{c=1}^R U_{ic} V_{jc} W_{kc} = 1$ for ALL i,j,k.
        
        # This is satisfied by the schoolbook with R = nmp.
        # Can we satisfy it with fewer?
        # Yes, if we use non-integer values or different structures.
        # But the task allows integer/half-integer.
        
        # My previous reasoning about Rank($2 \times 2 \times 2$)=7 relies on the fact that
        # $\sum_{c=1}^7 U_{ic} V_{jc} W_{kc} = 1$ for all i,j,k in {0,1}.
        # This is possible with Strassen's coefficients.
        
        # So I should implement the Strassen-based decomposition for the blocks.
        
        # Strassen for 2x2x2:
        # U, V, W are 4x7.
        # Indices: i in 0..1, j in 0..1, k in 0..1.
        # U[i, c], V[j, c], W[k, c].
        # We need $\sum_c U_{ic} V_{jc} W_{kc} = 1$.
        
        # Strassen coefficients (for 1s in the tensor):
        # The 7 products are linear combinations.
        # Actually, the standard Strassen decomposition for the **all-ones** tensor?
        # No, Strassen decomposes the **multiplication** tensor.
        # The multiplication tensor for 2x2x2 has $T_{ijk}=1$ for all i,j,k.
        # So yes, Strassen provides 7 vectors $u_c, v_c, w_c$ such that
        # $\sum_{c=1}^7 u_{ic} v_{jc} w_{kc} = 1$ for all i,j,k.
        
        # We can hardcode the Strassen vectors.
        
    # Strassen vectors for 2x2x2 (indices 0,1)
    # u: 2x7, v: 2x7, w: 2x7
    # Values can be 1, -1, 0.
    
    # Let's define the Strassen block for 2x2x2
    # u[0] = [1, 0, 1, 0, 1, 0, 0] ?
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # These are products of linear forms.
    # $u_c, v_c, w_c$ are the coefficients of the linear forms.
    # For M1:
    # $u_1 = [1, 1]^T$ (A11+A22)
    # $v_1 = [1, 1]^T$ (B11+B22)
    # $w_1 = [1, 1, 1, 1]^T$? No, w is for output.
    # The output is $C = \dots$
    # The tensor decomposition is $\sum u_c \otimes v_c \otimes w_c$.
    # The $w_c$ vector corresponds to how the product contributes to C.
    # For M1, it contributes to all 4 outputs?
    # $C_{11} = M1 - M4 - M3 + M7$
    # $C_{12} = M5 + M4$
    # $C_{21} = M2 + M3$
    # $C_{22} = M1 - M2 + M6 - M7$
    
    # So $w_1 = [1, 0, 0, 1]^T$? (Indices 0,1,2,3 for k? No, k is 0,1. Output is 2x2.
    # Wait, the tensor is 2x2x2. Output indices are i,k?
    # No, the tensor is $T_{ijk}$. i is row of A, j is col of A, k is col of B.
    # The decomposition is $\sum u_i v_j w_k$.
    # So $w$ is a vector of length 2 (for k=0,1).
    # But Strassen computes a 2x2 matrix C.
    # The "k" in the tensor $T_{ijk}$ is the column index of B.
    # The Strassen algorithm computes the whole 2x2 product.
    # So it decomposes the $2 \times 2 \times 2$ tensor (where the 3rd dim is the column of B?
    # No, the 3rd dim is the row of C?
    # The problem defines W[k*n+i, r].
    # So the 3rd index is k (column of B / row of C?)
    # $W[k*n+i]$. So the index is $k \cdot n + i$.
    # This implies the 3rd dimension of the tensor is size $p \cdot n$.
    # So the tensor is $n \times m \times (p \cdot n)$.
    # This is the standard "trace" convention.
    
    # So for 2x2x2, n=2, m=2, p=2.
    # W size is 4.
    # Strassen's 7 terms must sum to the identity in the appropriate sense.
    
    # This is getting complex to hardcode correctly without errors.
    # Given the 120s limit and risk of bugs, the schoolbook (84) is safe.
    # But 48 is a strong improvement if I can code it.
    
    # I will stick to a simpler improvement:
    # Just use the schoolbook but with a slightly better packing?
    # No, 84 is the baseline.
    
    # I will output the schoolbook to be safe, as coding Strassen for rectangular blocks
    # with the specific W-indexing is prone to off-by-one errors.
    
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
=======
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Use Strassen-based blocking to reduce rank.
    # Split m=4 into two blocks of size 2.
    # For each (n=3, m=2, p=7) block:
    #   Split n=3 into 1+2.
    #   (1,2,7): rank 2
    #   (2,2,7): split p=7 into 2+2+2+1
    #     Three (2,2,2) Strassen blocks: 3*7 = 21
    #     One (2,2,1) schoolbook: 2
    #     Total: 23
    #   Total for (3,2,7): 25
    # Two m-blocks: 50
    
    def strassen_222(U, V, W, u_off, v_off, w_off, start_col):
        """Add 7 Strassen terms for a 2x2x2 block.
        u_off: offset in U for i=0,1; j=0,1 -> U[u_off + i*2 + j]
        v_off: offset in V for j=0,1; k=0,1 -> V[v_off + j*2 + k]
        w_off: offset in W for k=0,1; i=0,1 -> W[w_off + k*2 + i]
        """
        # 7 Strassen terms
        # Each term: u_vec (len 4), v_vec (len 4), w_vec (len 4)
        # U index: i*2+j, V index: j*2+k, W index: k*2+i
        
        # M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1 -> [1,0,0,1]
        # v: B00=1, B11=1 -> [1,0,0,1]
        # w: C00=1, C11=1 -> W[0*2+0]=1, W[1*2+1]=1 -> [1,0,0,1]
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: [A00, A01, A10, A11], v: [B00, B01, B10, B11], w: [C00, C01, C10, C11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
            ([1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0]),   # M3
            ([0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]),    # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),    # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),   # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, -1]),  # M7
        ]
        
        for u_c, v_c, w_c in terms:
            col = start_col
            for idx_u, val_u in enumerate(u_c):
                if val_u != 0:
                    U[u_off + idx_u][col] += val_u
            for idx_v, val_v in enumerate(v_c):
                if val_v != 0:
                    V[v_off + idx_v][col] += val_v
            for idx_w, val_w in enumerate(w_c):
                if val_w != 0:
                    W[w_off + idx_w][col] += val_w
            start_col += 1
        return start_col
    
    def schoolbook_block(U, V, W, n_blk, m_blk, p_blk, i_off, j_off, k_off, start_col):
        """Add schoolbook terms for an n_blk x m_blk x p_blk block.
        U index: i_off + i*m_blk + j
        V index: j_off + j*p_blk + k
        W index: k_off + k*n_blk + i
        """
        for i in range(n_blk):
            for j in range(m_blk):
                for k in range(p_blk):
                    U[i_off + i * m_blk + j][start_col] = 1
                    V[j_off + j * p_blk + k][start_col] = 1
                    W[k_off + k * n_blk + i][start_col] = 1
                    start_col += 1
        return start_col
    
    # Total rank estimate
    total_rank = 0
    # Two m-blocks of size 2
    for m_blk_idx in range(2):
        # n=3, m=2, p=7
        # Split n into 1+2
        # Part 1: n=1, m=2, p=7 -> rank 2
        # Part 2: n=2, m=2, p=7 -> split p into 2+2+2+1
        #   3 Strassen (2,2,2) + 1 schoolbook (2,2,1) = 21 + 2 = 23
        total_rank += 2 + 23
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for m_blk_idx in range(2):
        j_off = m_blk_idx * 2  # j offset in V
        
        # Part 1: i=0 (n=1), j=j_off,j_off+1 (m=2), k=0..6 (p=7)
        # Schoolbook: 1*2*7 = 7 terms? No, for n=1, the tensor is T[0,j,k]=1
        # We need sum_c U[0*m+j,c] V[j*p+k,c] W[k*1+0,c] = 1
        # Schoolbook: c=(j,k), U[j,c]=1, V[j*p+k,c]=1, W[k,c]=1
        # That's 2*7=14 terms. But we can do better?
        # Actually for n=1, m=2, p=7: T[0,j,k]=1 for j in {0,1}, k in {0..6}
        # Schoolbook gives 14 terms. Can we do 2 terms?
        # If we use 2 terms: sum_{c=1}^2 U[j,c] V[j*p+k,c] W[k,c] = 1
        # For j=0,k=0: U[0,1]V[0,1]W[0,1] + U[0,2]V[0,2]W[0,2] = 1
        # For j=0,k=1: U[0,1]V[1,1]W[1,1] + U[0,2]V[1,2]W[1,2] = 1
        # ...
        # This is a system of 14 equations. With 2 terms, 12 unknowns (each term has 2+7+7=16 vars but shared)
        # Actually each term c has U[0,c],U[1,c] (2 vars), V[0*p+0..6,c] (7 vars), V[1*p+0..6,c] (7 vars), W[0..6,c] (7 vars)
        # Total 23 vars per term, 46 for 2 terms. 14 equations. Should be solvable.
        # But let's just use schoolbook for simplicity: 14 terms.
        # Wait, that would give total rank = 2*(14+23) = 74, worse than 84? No, 74 < 84.
        # But I estimated 25 per block. Let me recalculate.
        # For n=1,m=2,p=7: schoolbook is 1*2*7=14 terms.
        # For n=2,m=2,p=7: 3*7+2=23 terms.
        # Total per m-block: 14+23=37. Two blocks: 74.
        # That's better than 84 but not as good as my earlier estimate.
        
        # Let me use schoolbook for the n=1 part: 14 terms
        col = schoolbook_block(U, V, W, 1, 2, 7, 0, j_off, 0, col)
        
        # Part 2: i=1,2 (n=2), j=j_off,j_off+1 (m=2), k=0..6 (p=7)
        # Split k into 0,1 | 2,3 | 4,5 | 6
        k_offsets = [0, 2, 4, 6]
        k_sizes = [2, 2, 2, 1]
        
        for k_idx in range(4):
            k_off = k_offsets[k_idx]
            k_sz = k_sizes[k_idx]
            i_off = 1  # i starts at 1
            # U index: i_off + i*2 + j, where i in {0,1} relative, so actual i = i_off + i
            # V index: j_off + j*2 + k, where j in {0,1}, k in {k_off..k_off+k_sz-1}
            # W index: k_off + k*2 + i, where k in {0..k_sz-1} relative, i in {0,1}
            
            if k_sz == 2:
                col = strassen_222(U, V, W, i_off * 2 + 0, j_off, k_off, col)
            else:
                # k_sz == 1: schoolbook 2*2*1 = 4 terms
                col = schoolbook_block(U, V, W, 2, 2, 1, i_off, j_off, k_off, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize my indexing in `strassen_222` is wrong. The U offset should account for the full U index which is `i*m + j` where m=4 (the full m), not m_blk=2. Let me fix this.

Also, I need to verify the Strassen coefficients are correct. Let me rewrite more carefully.


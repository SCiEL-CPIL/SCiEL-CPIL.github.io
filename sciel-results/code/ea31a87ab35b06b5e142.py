import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 4
    
    # Build rank-49 Strassen^2 decomposition
    # 2x2 Strassen factors
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    rank = 49
    U_arr = np.zeros((16, rank), dtype=int)
    V_arr = np.zeros((16, rank), dtype=int)
    W_arr = np.zeros((16, rank), dtype=int)
    
    # Mapping for 4x4 indices to 2x2 block indices and 2x2 inner indices
    # Index i in 0..3 -> block_row = i//2, inner_row = i%2
    # Index j in 0..3 -> block_col = j//2, inner_col = j%2
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # Fill U_arr
            # U_arr[i*4 + j, r] where i,j are 0..3
            # U2 is 2x2.
            # A[i,j] is split into A[block_row, block_col] which is a 2x2 matrix.
            # The term in U2 for block (br, bc) is U2[br*2 + br_inner, s]? 
            # No, U2 is for the 2x2 matrix multiplication of blocks.
            # Let A be 2x2 blocks of 2x2.
            # A_ik (block) = sum_{a,b} A_{i*2+a, k*2+b}
            # The Strassen decomposition acts on the block indices.
            # So for a fixed Strassen term (s,t), the coefficient for A_{i,j} is:
            # U2[ (i//2)*2 + (i%2) ??? No.
            
            # Let's trace:
            # C = A B
            # A is 4x4. Viewed as 2x2 blocks of 2x2.
            # A_{I,J} (block) is a 2x2 matrix.
            # C_{I,K} = sum_J A_{I,J} B_{J,K}
            # Each A_{I,J} is 2x2.
            # The multiplication of 2x2 matrices A_{I,J} * B_{J,K} is done via Strassen?
            # No, the "outer" Strassen decomposes the block multiplication.
            # C_{I,K} = sum_J M^{(J)}_{I,K}
            # Where M^{(J)} are intermediate 2x2 matrices.
            # Each M is a 2x2 matrix.
            # The computation of M involves linear combinations of A_{I,J} and B_{J,K}.
            # Then the final C_{I,K} is a sum of these M's.
            
            # The current code logic:
            # U_arr[i*4 + j, r] = U2[ I*2 + i_inner, s ] * U2[ i_inner2, t ]? 
            # Let's look at the existing code's logic which worked (score 1.3061).
            
            for i in range(4):
                for j in range(4):
                    # i is row of A, j is col of A
                    # Block row: i//2, Block col: j//2
                    # Inner row: i%2, Inner col: j%2
                    # The factor U2 is for the block-level multiplication.
                    # The block A_{i//2, j//2} contributes to the s-th Strassen term.
                    # The inner 2x2 multiplication is handled by the t-th Strassen term?
                    # No, the tensor is $T_{i,j,a} = \delta_{i,a} \delta_{j,?}$?
                    # The tensor for 4x4 is the Kronecker product of the 2x2 tensor with itself?
                    # $T_4 = T_2 \otimes T_2$.
                    # The rank of $T_2 \otimes T_2$ is at most $rank(T_2) \times rank(T_2) = 49$.
                    # The decomposition is:
                    # $U_4 = U_2 \otimes U_2$
                    # $V_4 = V_2 \otimes V_2$
                    # $W_4 = W_2 \otimes W_2$
                    # (With appropriate reshaping/indexing).
                    
                    # Index in U_2: $u_2 = (i//2)*2 + (i%2)$? No.
                    # $U_2$ has shape (4, 7). Indices 0..3 correspond to elements of 2x2 matrix.
                    # Let's say index $k$ in $U_2$ corresponds to $A_{r,c}$ with $r=k//2, c=k%2$.
                    # So for $A_{i,j}$, the block index is $I=i//2, J=j//2$.
                    # The inner index is $i_{in}=i%2, j_{in}=j%2$.
                    # The element $A_{i,j}$ is at position $I*2 + i_{in}$ in the row?
                    # No, the 2x2 matrix $A_{I,J}$ has elements $A_{I*2+i_{in}, J*2+j_{in}}$.
                    # So the "flattened" index of $A_{I,J}$'s element $(i_{in}, j_{in})$ is $I*2 + i_{in}$ for row?
                    # Wait, $U_2$ is for the 2x2 matrix multiplication.
                    # The input to $U_2$ is a 2x2 matrix.
                    # So $U_2$ maps the 4 elements of the 2x2 matrix to the 7 Strassen intermediates.
                    # So $U_2[k, s]$ where $k$ is the index of the element in the 2x2 matrix.
                    # For $A_{i,j}$, the block is $A_{I,J}$.
                    # The element is $A_{I,J}[i_{in}, j_{in}]$.
                    # In the flattened 2x2 matrix $A_{I,J}$, the index is $i_{in}*2 + j_{in}$.
                    # So we need $U_2[i_{in}*2 + j_{in}, s]$?
                    # No, $U_2$ is applied to the block $A_{I,J}$.
                    # So the coefficient for $A_{i,j}$ in the s-th term of the "outer" Strassen is $U_2[ \text{index of } A_{I,J} \text{ in the block matrix?} ]$?
                    # No. The "outer" Strassen operates on the **blocks**.
                    # So $U_2$ should be applied to the **block indices**.
                    # But $U_2$ has 4 rows, corresponding to the 4 elements of a 2x2 matrix.
                    # This implies that the "outer" Strassen is treating the 2x2 **blocks** as scalars?
                    # No, that would give rank 7 for the block multiplication, but each block multiplication is a 2x2 matmul.
                    # So we need to do Strassen for the blocks, AND Strassen for the inner elements.
                    # This is the Kronecker product structure.
                    
                    # The term $r = s*7 + t$.
                    # $s$ corresponds to the outer Strassen term (block level).
                    # $t$ corresponds to the inner Strassen term (element level).
                    
                    # For $A_{i,j}$:
                    # Block index: $I = i//2, J = j//2$.
                    # Inner index: $i_{in} = i%2, j_{in} = j%2$.
                    # The contribution to the s-th outer term is determined by $U_2$ applied to the block matrix.
                    # But $U_2$ is a 4x7 matrix. Its rows correspond to the elements of the 2x2 **block matrix**?
                    # No, the 2x2 block matrix has 4 blocks.
                    # So $U_2$'s rows 0..3 correspond to blocks $A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1}$.
                    # So $U_2[I*2+J, s]$?
                    # And the inner part: $U_2[i_{in}*2+j_{in}, t]$?
                    # Let's check the existing code:
                    # `coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]`
                    # `I_map[i] = i//2`, `J_map[j] = j//2`.
                    # So `U2[(i//2)*2 + (j//2), s]`.
                    # And `U2[(i%2)*2 + (j%2), t]`.
                    # This matches the Kronecker product logic.
                    
                    I = i // 2
                    J = j // 2
                    ii = i % 2
                    jj = j % 2
                    coeff = U2[I*2 + J, s] * U2[ii*2 + jj, t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
                        
            for k in range(4):
                for l in range(4):
                    I = k // 2
                    J = l // 2
                    ii = k % 2
                    jj = l % 2
                    coeff = V2[I*2 + J, s] * V2[ii*2 + jj, t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
                        
            for j in range(4):
                for i in range(4):
                    # W index convention: W[k*n+i, r]
                    # k is col of C (j), i is row of C (i).
                    # So index is j*4 + i.
                    # Block row: i//2, Block col: j//2.
                    # Inner row: i%2, Inner col: j%2.
                    I = i // 2
                    J = j // 2
                    ii = i % 2
                    jj = j % 2
                    coeff = W2[I*2 + J, s] * W2[ii*2 + jj, t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff
                        
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

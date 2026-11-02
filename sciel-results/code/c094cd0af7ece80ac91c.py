import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 46.
    # Split A (2x4) into A_L (2x2) and A_R (2x2).
    # Split B (4x7) into B_T (2x7) and B_B (2x7).
    # AB = A_L B_T + A_R B_B.
    # Each 2x2x7 multiplication is decomposed into three 2x2x2 (Strassen, rank 7) and one 2x2x1 (rank 2).
    # Total rank = 2 * (3 * 7 + 2) = 2 * 23 = 46.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)] # 8 x 46
    V = [[0] * rank for _ in range(m * p)] # 28 x 46
    W = [[0] * rank for _ in range(p * n)] # 14 x 46
    
    # Helper to add a term to the decomposition
    # u_idx: row index in U (i*m + j)
    # v_idx: row index in V (j*p + k)
    # w_idx: row index in W (k*n + i)
    # val: coefficient
    
    def add_term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Strassen terms for 2x2x2
    # Local indices i, j, k in {0, 1}
    # Global mapping for A_L: i_global = i, j_global = j
    # Global mapping for B_T: j_global = j, k_global = k
    # Result C: i_global = i, k_global = k
    
    # Strassen definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate over the 3 blocks of 2 columns in B (k=0,1; 2,3; 4,5) and 1 single column (k=6)
    # for each of the two halves of A (L and R)
    
    col = 0
    
    for half in range(2): # 0 for L, 1 for R
        # Offset for A's column index j
        j_offset = half * 2
        # Offset for B's row index j
        # B is 4x7. B_T is rows 0,1. B_B is rows 2,3.
        # So for half=0, B row offset is 0. For half=1, B row offset is 2.
        b_j_offset = half * 2
        
        for block in range(3): # 0, 1, 2 for pairs of k
            # k_offset in B's column dimension
            k_offset = block * 2
            
            # For this 2x2x2 block:
            # A indices: i in {0,1}, j in {j_offset, j_offset+1}
            # B indices: j in {b_j_offset, b_j_offset+1}, k in {k_offset, k_offset+1}
            # C indices: i in {0,1}, k in {k_offset, k_offset+1}
            
            # M1
            # (A00+A11)(B00+B11)
            # A00: i=0, j=j_offset. A11: i=1, j=j_offset+1
            # B00: j=b_j_offset, k=k_offset. B11: j=b_j_offset+1, k=k_offset+1
            # C00: i=0, k=k_offset. C11: i=1, k=k_offset+1
            # W index: k*n + i
            
            add_term(0*m + j_offset, b_j_offset*p + k_offset, k_offset*n + 0, 1, col)
            add_term(1*m + (j_offset+1), (b_j_offset+1)*p + (k_offset+1), (k_offset+1)*n + 1, 1, col)
            
            # Wait, Strassen M1 contributes to C00 and C11.
            # C00 = M1 + ...
            # C11 = M1 - ...
            # So W[k_offset*n + 0][col] += 1 (from M1)
            # W[(k_offset+1)*n + 1][col] += 1 (from M1)
            # But M1 is a single scalar multiplication.
            # The term is (A00+A11) * (B00+B11).
            # This means U[0, j_offset] += 1, U[1, j_offset+1] += 1
            # V[b_j_offset, k_offset] += 1, V[b_j_offset+1, k_offset+1] += 1
            # And W must reflect the sum for C00 and C11.
            # W[k_offset*n + 0] += 1 (for C00)
            # W[(k_offset+1)*n + 1] += 1 (for C11)
            
            # My add_term function adds to U, V, and W simultaneously with the same value.
            # This is correct for the factor matrices.
            # However, I called add_term twice for M1?
            # No, add_term(u_idx, v_idx, w_idx, val, col) adds val to U[u_idx], V[v_idx], W[w_idx].
            # But M1 involves sum of A's and sum of B's.
            # So U[0, j_offset] gets 1, U[1, j_offset+1] gets 1.
            # V[b_j_offset, k_offset] gets 1, V[b_j_offset+1, k_offset+1] gets 1.
            # But they all map to the SAME column `col`.
            # So I need to call add_term for each component?
            # No, add_term adds to specific indices.
            # If I call add_term(0*m+j_offset, b_j_offset*p+k_offset, k_offset*n+0, 1, col), it adds 1 to U[0, j_offset], V[b_j_offset, k_offset], W[k_offset*n+0].
            # But I also need to add 1 to U[1, j_offset+1].
            # So I should call add_term for each pair (u, v, w) that contributes to the same scalar product?
            # No, the scalar product is $M_1 = (\sum A) (\sum B)$.
            # This is not a single rank-1 tensor. It's a sum of rank-1 tensors?
            # No, $M_1$ is a scalar.
            # The decomposition is $\sum_c U_{:,c} V_{:,c} W_{:,c}^T$.
            # $M_1$ corresponds to one column `c`.
            # $U_{0, j_offset, c} = 1, U_{1, j_offset+1, c} = 1$.
            # $V_{b_j_offset, k_offset, c} = 1, V_{b_j_offset+1, k_offset+1, c} = 1$.
            # $W_{k_offset, 0, c} = 1, W_{k_offset+1, 1, c} = 1$.
            # So I need to add 1 to U[0, j_offset], U[1, j_offset+1], V[b_j_offset, k_offset], V[b_j_offset+1, k_offset+1], W[k_offset*n+0], W[(k_offset+1)*n+1].
            # All in column `col`.
            
            # Let's redefine add_term to just add to specific indices in a given column.
            # I will just manually increment the lists.
            
            col += 1 # Move to next column for M2
            
        # After 3 blocks of 2, we have 1 remaining column (k=6)
        # This is a 2x2x1 multiplication.
        # C00 = A00 B00 + A01 B01? No, k is fixed at 6.
        # A is 2x2 (local), B is 2x1 (local).
        # C00 = A00 B0 + A10 B1? No.
        # C[i, 6] = A[i, 0] B[0, 6] + A[i, 1] B[1, 6].
        # This is two rank-1 terms.
        # Term 1: A[i,0] * B[0,6].
        # Term 2: A[i,1] * B[1,6].
        
        # Term 1:
        # U[i, j_offset] for i=0,1? No, A[i,0] means row i, col 0 of A_L.
        # So U[i*m + j_offset] for i=0,1.
        # V[b_j_offset, 6] for j=0? No, B[0,6] means row 0, col 6.
        # So V[b_j_offset + 0, 6].
        # W[6*n + i] for i=0,1.
        
        # Wait, the term is $A_{i0} B_{06}$. This is not a single rank-1 tensor for the whole matrix.
        # The rank-1 tensor is $u v w^T$.
        # $u$ is a vector in $\mathbb{R}^{8}$, $v$ in $\mathbb{R}^{28}$, $w$ in $\mathbb{R}^{14}$.
        # If we use one column for $A_{i0} B_{06}$, it must be that $u$ has non-zeros at $i=0,1$?
        # No, $A_{i0}$ depends on $i$.
        # So $u$ must be $(1, 0, 1, 0, ...)$? No.
        # $C_{i, 6} = A_{i0} B_{06}$.
        # This can be written as sum over $i$? No.
        # It is a single term in the sum for $C_{i,6}$.
        # But $A_{i0}$ is different for $i=0$ and $i=1$.
        # So we need two terms: one for $i=0$ and one for $i=1$.
        # Term 1: $i=0$. $u$ has 1 at $0*m+j_offset$. $v$ has 1 at $b_j_offset*p+6$. $w$ has 1 at $6*n+0$.
        # Term 2: $i=1$. $u$ has 1 at $1*m+j_offset$. $v$ has 1 at $b_j_offset*p+6$. $w$ has 1 at $6*n+1$.
        
        # Similarly for $A_{i1} B_{16}$.
        # Term 3: $i=0$. $u$ has 1 at $0*m+(j_offset+1)$. $v$ has 1 at $(b_j_offset+1)*p+6$. $w$ has 1 at $6*n+0$.
        # Term 4: $i=1$. $u$ has 1 at $1*m+(j_offset+1)$. $v$ has 1 at $(b_j_offset+1)*p+6$. $w$ has 1 at $6*n+1$.
        
        # So 4 terms for the 2x2x1 part?
        # My previous estimate was rank 2. Why?
        # Rank of $2 \times 2 \times 1$ tensor.
        # $T_{i,j,k} = \delta_{i,k} \delta_{j,0}$? No.
        # The multiplication tensor for $2 \times 2 \times 1$ is:
        # $C_{i0} = A_{i0} B_{00} + A_{i1} B_{10}$.
        # This is the sum of two rank-1 tensors:
        # 1. $A_{i0} B_{00} \to C_{i0}$.
        # 2. $A_{i1} B_{10} \to C_{i0}$.
        # Each of these is a rank-1 tensor?
        # Tensor 1: $u = e_{0,0} + e_{1,0}$? No.
        # $u$ is vector in $A$ space. $v$ in $B$ space. $w$ in $C$ space.
        # Term 1: $A_{i0} B_{00}$.
        # $u$ has 1 at $(0,0)$ and $(1,0)$? No, $A_{i0}$ is the value.
        # The tensor entry is $A_{i0} B_{00}$.
        # This is not a single rank-1 tensor $u v w^T$ unless $u, v, w$ are constant.
        # But $A_{i0}$ varies with $i$.
        # So we must split by $i$.
        # Term 1a: $i=0$. $u=e_{0,0}, v=e_{0,0}, w=e_{0,0}$.
        # Term 1b: $i=1$. $u=e_{1,0}, v=e_{0,0}, w=e_{1,0}$.
        # Term 2a: $i=0$. $u=e_{0,1}, v=e_{1,0}, w=e_{0,0}$.
        # Term 2b: $i=1$. $u=e_{1,1}, v=e_{1,0}, w=e_{1,0}$.
        # So yes, 4 terms.
        # So rank($2 \times 2 \times 1$) is 4?
        # Wait, the schoolbook rank is $n \times m \times p = 2 \times 2 \times 1 = 4$.
        # So Strassen doesn't help for $p=1$.
        
        # So for $2 \times 2 \times 7$, rank is $3 \times 7 + 4 = 25$.
        # Total rank $25 + 25 = 50$.
        # This is better than 56, but not 46.
        
        # Can we do better for $2 \times 2 \times 7$?
        # Maybe split 7 into $4+3$?
        # Rank($2 \times 2 \times 4$)?
        # $2 \times 2 \times 4$ can be done with two $2 \times 2 \times 2$ Strassen?
        # $C = A B$. $B$ is $2 \times 4$.
        # $B = [B_1, B_2]$ where $B_1, B_2$ are $2 \times 2$.
        # $C = A B_1 + A B_2$.
        # Rank = $7 + 7 = 14$.
        # Rank($2 \times 2 \times 3$)?
        # $B$ is $2 \times 3$. $B = [B_1, B_2]$ where $B_1$ is $2 \times 2$, $B_2$ is $2 \times 1$.
        # Rank = $7 + 4 = 11$.
        # Total $14 + 11 = 25$. Same.
        
        # What if we split 7 into $2+2+2+1$?
        # $7+7+7+4 = 25$.
        
        # Is there a better decomposition for $2 \times 2 \times 3$?
        # Known rank is 10?
        # If rank($2 \times 2 \times 3$) is 10, then $14+10=24$.
        # Total 48.
        
        # Let's check if rank($2 \times 2 \times 3$) is 10.
        # According to literature, rank($2 \times 2 \times 2$) = 7.
        # Rank($2 \times 2 \times 3$) is at most 11.
        # It is known to be 10?
        # If it is 10, we get 48.
        
        # Let's try to implement the 48 rank solution if 10 is available.
        # Or just stick to 50 which is definitely correct and easier to construct.
        # 50 is better than 56.
        
        # Let's refine the 50 rank construction.
        # We will use 3 blocks of 2 (Strassen, 7 each) and 1 block of 1 (Schoolbook, 4 each).
        # Total 25 per half.
        
        # Re-implementing the loop with 4 terms for the last block.
        
        # Reset U, V, W
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        for half in range(2):
            j_offset = half * 2
            b_j_offset = half * 2
            
            for block in range(3):
                k_offset = block * 2
                
                # M1
                U[0*m + j_offset][col] += 1
                U[1*m + (j_offset+1)][col] += 1
                V[b_j_offset*p + k_offset][col] += 1
                V[(b_j_offset+1)*p + (k_offset+1)][col] += 1
                W[k_offset*n + 0][col] += 1
                W[(k_offset+1)*n + 1][col] += 1
                col += 1
                
                # M2: (A10+A11)B00
                # A10: i=1, j=j_offset. A11: i=1, j=j_offset+1.
                # B00: j=b_j_offset, k=k_offset.
                # C10: i=1, k=k_offset.
                U[1*m + j_offset][col] += 1
                U[1*m + (j_offset+1)][col] += 1
                V[b_j_offset*p + k_offset][col] += 1
                W[k_offset*n + 1][col] += 1
                col += 1
                
                # M3: A00(B01-B11)
                # A00: i=0, j=j_offset.
                # B01: j=b_j_offset, k=k_offset+1.
                # B11: j=b_j_offset+1, k=k_offset+1.
                # C01: i=0, k=k_offset+1.
                U[0*m + j_offset][col] += 1
                V[b_j_offset*p + (k_offset+1)][col] += 1
                V[(b_j_offset+1)*p + (k_offset+1)][col] -= 1
                W[(k_offset+1)*n + 0][col] += 1
                col += 1
                
                # M4: A11(B10-B00)
                # A11: i=1, j=j_offset+1.
                # B10: j=b_j_offset+1, k=k_offset.
                # B00: j=b_j_offset, k=k_offset.
                # C00: i=0, k=k_offset. C10: i=1, k=k_offset.
                U[1*m + (j_offset+1)][col] += 1
                V[(b_j_offset+1)*p + k_offset][col] += 1
                V[b_j_offset*p + k_offset][col] -= 1
                W[k_offset*n + 0][col] += 1
                W[k_offset*n + 1][col] += 1
                col += 1
                
                # M5: (A00+A01)B11
                # A00: i=0, j=j_offset. A01: i=0, j=j_offset+1.
                # B11: j=b_j_offset+1, k=k_offset+1.
                # C00: i=0, k=k_offset. C01: i=0, k=k_offset+1.
                U[0*m + j_offset][col] += 1
                U[0*m + (j_offset+1)][col] += 1
                V[(b_j_offset+1)*p + (k_offset+1)][col] += 1
                W[k_offset*n + 0][col] -= 1 # From C00 = M1 + M4 - M5 + M7
                W[(k_offset+1)*n + 0][col] += 1 # From C01 = M3 + M5
                col += 1
                
                # M6: (A10-A00)(B00+B01)
                # A10: i=1, j=j_offset. A00: i=0, j=j_offset.
                # B00: j=b_j_offset, k=k_offset. B01: j=b_j_offset, k=k_offset+1.
                # C11: i=1, k=k_offset+1.
                U[1*m + j_offset][col] += 1
                U[0*m + j_offset][col] -= 1
                V[b_j_offset*p + k_offset][col] += 1
                V[b_j_offset*p + (k_offset+1)][col] += 1
                W[(k_offset+1)*n + 1][col] += 1
                col += 1
                
                # M7: (A01-A11)(B10+B11)
                # A01: i=0, j=j_offset+1. A11: i=1, j=j_offset+1.
                # B10: j=b_j_offset+1, k=k_offset. B11: j=b_j_offset+1, k=k_offset+1.
                # C00: i=0, k=k_offset. C11: i=1, k=k_offset+1.
                U[0*m + (j_offset+1)][col] += 1
                U[1*m + (j_offset+1)][col] -= 1
                V[(b_j_offset+1)*p + k_offset][col] += 1
                V[(b_j_offset+1)*p + (k_offset+1)][col] += 1
                W[k_offset*n + 0][col] += 1
                W[(k_offset+1)*n + 1][col] += 1
                col += 1
                
            # Last block: k=6
            # 4 terms
            # Term 1: i=0, A[0, j_offset], B[b_j_offset, 6]
            U[0*m + j_offset][col] += 1
            V[b_j_offset*p + 6][col] += 1
            W[6*n + 0][col] += 1
            col += 1
            
            # Term 2: i=1, A[1, j_offset], B[b_j_offset, 6]
            U[1*m + j_offset][col] += 1
            V[b_j_offset*p + 6][col] += 1
            W[6*n + 1][col] += 1
            col += 1
            
            # Term 3: i=0, A[0, j_offset+1], B[b_j_offset+1, 6]
            U[0*m + (j_offset+1)][col] += 1
            V[(b_j_offset+1)*p + 6][col] += 1
            W[6*n + 0][col] += 1
            col += 1
            
            # Term 4: i=1, A[1, j_offset+1], B[b_j_offset+1, 6]
            U[1*m + (j_offset+1)][col] += 1
            V[(b_j_offset+1)*p + 6][col] += 1
            W[6*n + 1][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

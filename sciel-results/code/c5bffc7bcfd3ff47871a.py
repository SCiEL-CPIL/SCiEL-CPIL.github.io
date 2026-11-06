import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition 5 into 2, 2, 1
    sizes = [2, 2, 1]
    
    # Precompute start indices for each block dimension
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
        
    # Total rank is sum of ranks of each block
    # Block (i, j, k) has size sizes[i] x sizes[j] x sizes[k]
    # Rank is 7 if all sizes are 2, else product of sizes
    
    rank_blocks = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = sizes[i], sizes[j], sizes[k]
                if si == 2 and sj == 2 and sk == 2:
                    rank_blocks.append(7)
                else:
                    rank_blocks.append(si * sj * sk)
                    
    total_rank = sum(rank_blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Strassen 2x2x2 patterns
    # Each pattern is a list of (u_indices, v_indices, w_indices, value)
    # u_indices: list of (local_i, local_j)
    # v_indices: list of (local_j, local_k)
    # w_indices: list of (local_k, local_i)
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    strassen_1 = [
        ([ (0,0), (1,1) ], [ (0,0), (1,1) ], [ (0,0), (1,1) ], 1),
    ]
    
    # M2 = (A10+A11)B00
    # C10 += M2
    strassen_2 = [
        ([ (1,0), (1,1) ], [ (0,0) ], [ (0,1) ], 1),
    ]
    
    # M3 = A00(B01-B11)
    # C01 += M3
    strassen_3 = [
        ([ (0,0) ], [ (0,1) ], [ (1,0) ], 1),
        ([ (0,0) ], [ (1,1) ], [ (1,0) ], -1),
    ]
    
    # M4 = A11(B10-B00)
    # C10 += M4, C00 += M4
    strassen_4 = [
        ([ (1,1) ], [ (1,0) ], [ (0,1) ], 1),
        ([ (1,1) ], [ (1,0) ], [ (1,1) ], 1), # Wait, C11? No.
        # C10 gets M4. C00 gets M4.
        # C10: k=0, i=1 -> W[0,1]
        # C00: k=0, i=0 -> W[0,0]
        # My previous deduction:
        # Term 2: C10 += M2. W[0,1].
        # Term 4: C10 += M4, C00 += M4.
        # So W[0,1] and W[0,0].
        ([ (1,1) ], [ (0,0) ], [ (0,1) ], -1),
        ([ (1,1) ], [ (0,0) ], [ (0,0) ], -1),
    ]
    # Let's re-verify Strassen carefully.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Local indices 0,1 correspond to 1,2 in 1-based.
    # A00=A11, A01=A12, A10=A21, A11=A22.
    # B00=B11, B01=B12, B10=B21, B11=B22.
    # C00=C11, C01=C12, C10=C21, C11=C22.
    
    # M1: A00+A11, B00+B11. C00+=M1, C11+=M1.
    # U: (0,0), (1,1). V: (0,0), (1,1). W: (0,0), (1,1).
    
    # M2: A10+A11, B00. C10+=M2.
    # U: (1,0), (1,1). V: (0,0). W: (0,1).
    
    # M3: A00, B01-B11. C01+=M3.
    # U: (0,0). V: (0,1), (1,1). W: (1,0).
    # Note: B01 has coeff 1, B11 has coeff -1.
    # So two terms:
    # 1. U:(0,0), V:(0,1), W:(1,0), val 1
    # 2. U:(0,0), V:(1,1), W:(1,0), val -1
    
    # M4: A11, B10-B00. C10+=M4, C00+=M4.
    # U: (1,1). V: (1,0), (0,0). W: (0,1), (0,0).
    # B10 coeff 1, B00 coeff -1.
    # C10 (k=0, i=1) -> W(0,1).
    # C00 (k=0, i=0) -> W(0,0).
    # Terms:
    # 1. U:(1,1), V:(1,0), W:(0,1), val 1
    # 2. U:(1,1), V:(1,0), W:(0,0), val 1
    # 3. U:(1,1), V:(0,0), W:(0,1), val -1
    # 4. U:(1,1), V:(0,0), W:(0,0), val -1
    
    # M5: A00+A01, B11. C00-=M5, C01+=M5.
    # U: (0,0), (0,1). V: (1,1). W: (0,0), (1,0).
    # C00 (k=0, i=0) -> W(0,0).
    # C01 (k=1, i=0) -> W(1,0).
    # Terms:
    # 1. U:(0,0), V:(1,1), W:(0,0), val -1
    # 2. U:(0,0), V:(1,1), W:(1,0), val 1
    # 3. U:(0,1), V:(1,1), W:(0,0), val -1
    # 4. U:(0,1), V:(1,1), W:(1,0), val 1
    
    # M6: A10-A00, B00+B01. C11+=M6.
    # U: (1,0), (0,0). V: (0,0), (0,1). W: (1,1).
    # C11 (k=1, i=1) -> W(1,1).
    # Terms:
    # 1. U:(1,0), V:(0,0), W:(1,1), val 1
    # 2. U:(1,0), V:(0,1), W:(1,1), val 1
    # 3. U:(0,0), V:(0,0), W:(1,1), val -1
    # 4. U:(0,0), V:(0,1), W:(1,1), val -1
    
    # M7: A01-A11, B10+B11. C00+=M7, C11+=M7.
    # U: (0,1), (1,1). V: (1,0), (1,1). W: (0,0), (1,1).
    # C00 (k=0, i=0) -> W(0,0).
    # C11 (k=1, i=1) -> W(1,1).
    # Terms:
    # 1. U:(0,1), V:(1,0), W:(0,0), val 1
    # 2. U:(0,1), V:(1,1), W:(0,0), val 1
    # 3. U:(0,1), V:(1,0), W:(1,1), val 1
    # 4. U:(0,1), V:(1,1), W:(1,1), val 1
    # 5. U:(1,1), V:(1,0), W:(0,0), val -1
    # 6. U:(1,1), V:(1,1), W:(0,0), val -1
    # 7. U:(1,1), V:(1,0), W:(1,1), val -1
    # 8. U:(1,1), V:(1,1), W:(1,1), val -1
    
    # To minimize rank, we can combine terms if they share U,V,W patterns?
    # No, Strassen is already minimal for 2x2x2. The 7 "multiplications" are the rank.
    # Each multiplication corresponds to a column in U, V, W.
    # However, the expansion above shows that one "multiplication" (like M1) involves
    # multiple non-zero entries in U, V, W for that single column.
    # So for M1, we add ONE column to U, V, W.
    # U_col has 1 at (0,0) and (1,1).
    # V_col has 1 at (0,0) and (1,1).
    # W_col has 1 at (0,0) and (1,1).
    
    # This is correct. The rank is the number of columns.
    
    strassen_terms = [
        # M1
        ( [ (0,0), (1,1) ], [ (0,0), (1,1) ], [ (0,0), (1,1) ], 1 ),
        # M2
        ( [ (1,0), (1,1) ], [ (0,0) ], [ (0,1) ], 1 ),
        # M3 (split into 2 cols? No, M3 is one multiplication A00 * (B01-B11).
        # The factor (B01-B11) is linear.
        # So U is A00. V is B01 and -B11. W is C01.
        # This is ONE column in the decomposition.
        # U: (0,0). V: (0,1) val 1, (1,1) val -1. W: (1,0) val 1.
        ( [ (0,0) ], [ (0,1), (1,1) ], [ (1,0) ], [1, -1, 1] ), # Note: V has 2 entries, W has 1.
        # Wait, the values must correspond.
        # U[0,0]=1.
        # V[0,1]=1, V[1,1]=-1.
        # W[1,0]=1.
        # This is one column.
        
        # M4
        ( [ (1,1) ], [ (1,0), (0,0) ], [ (0,1), (0,0) ], [1, -1, 1, 1] ),
        # U[1,1]=1.
        # V[1,0]=1, V[0,0]=-1.
        # W[0,1]=1, W[0,0]=1.
        
        # M5
        ( [ (0,0), (0,1) ], [ (1,1) ], [ (0,0), (1,0) ], [-1, 1, -1, 1] ),
        # U[0,0]=-1, U[0,1]=1.
        # V[1,1]=1.
        # W[0,0]=-1, W[1,0]=1.
        
        # M6
        ( [ (1,0), (0,0) ], [ (0,0), (0,1) ], [ (1,1) ], [1, -1, 1, -1] ),
        # U[1,0]=1, U[0,0]=-1.
        # V[0,0]=1, V[0,1]=1.
        # W[1,1]=1.
        
        # M7
        ( [ (0,1), (1,1) ], [ (1,0), (1,1) ], [ (0,0), (1,1) ], [1, -1, 1, -1, 1, -1, 1, -1] ),
        # U[0,1]=1, U[1,1]=-1.
        # V[1,0]=1, V[1,1]=1.
        # W[0,0]=1, W[1,1]=1.
    ]
    
    # Helper to add a term
    def add_term(U, V, W, i_start, j_start, k_start, u_list, v_list, w_list, vals):
        # u_list, v_list, w_list are lists of (local_i, local_j) etc.
        # vals is a list of values corresponding to the entries.
        # We need to map local to global.
        # The number of entries in u_list, v_list, w_list must match the number of values?
        # No, the values apply to the specific non-zeros.
        # Actually, it's easier to just iterate and set values.
        # But wait, if U has 2 entries and V has 2 entries, the product is 4 terms?
        # No, in tensor decomposition, a column r defines:
        # U[i,j,r] = u_val
        # V[j,k,r] = v_val
        # W[k,i,r] = w_val
        # The contribution to C[i,k] is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # For a fixed r, if U has non-zeros at (i1,j1), (i2,j2) and V at (j1,k1), (j2,k2)...
        # It doesn't "multiply" them within the column.
        # The column r represents a rank-1 tensor.
        # A rank-1 tensor is $u_i v_j w_k$.
        # If $u$ has support at $i_1, i_2$, it's not rank 1 in the subspace?
        # Yes, a vector with multiple non-zeros is fine. It's still a vector.
        # So for one column r:
        # U[i,j,r] is non-zero for a set of (i,j).
        # V[j,k,r] is non-zero for a set of (j,k).
        # W[k,i,r] is non-zero for a set of (k,i).
        # The product $U[i,j,r] V[j,k,r] W[k,i,r]$ is non-zero only if there exists
        # a path $i \to j \to k \to i$ in the support?
        # No, it's just the product of the three values for the SAME i,j,k.
        # So if $U[0,0,r]=1, U[1,1,r]=1$.
        # And $V[0,0,r]=1, V[1,1,r]=1$.
        # And $W[0,0,r]=1, W[1,1,r]=1$.
        # Then for $i=0,j=0,k=0$: $1\cdot1\cdot1 = 1$.
        # For $i=1,j=1,k=1$: $1\cdot1\cdot1 = 1$.
        # For $i=0,j=1,k=0$: $U[0,1]=0 \implies 0$.
        # This correctly generates the diagonal terms of M1.
        
        # So I just need to populate the arrays.
        pass

    # Implementing the loop
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                si, sj, sk = sizes[i_block], sizes[j_block], sizes[k_block]
                i_start = starts[i_block]
                j_start = starts[j_block]
                k_start = starts[k_block]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen
                    for u_list, v_list, w_list, vals in strassen_terms:
                        # This structure is tricky because vals length depends on total non-zeros?
                        # Let's just hardcode the 7 columns explicitly for clarity and correctness.
                        pass
                else:
                    # Schoolbook
                    pass
                
                col_offset += len(rank_blocks) # Placeholder

    # Refining the implementation for speed and correctness
    # I will construct the 7 Strassen columns explicitly.
    
    # M1
    u_1 = [(0,0,1), (1,1,1)]
    v_1 = [(0,0,1), (1,1,1)]
    w_1 = [(0,0,1), (1,1,1)]
    
    # M2
    u_2 = [(1,0,1), (1,1,1)]
    v_2 = [(0,0,1)]
    w_2 = [(0,1,1)]
    
    # M3
    u_3 = [(0,0,1)]
    v_3 = [(0,1,1), (1,1,-1)]
    w_3 = [(1,0,1)]
    
    # M4
    u_4 = [(1,1,1)]
    v_4 = [(1,0,1), (0,0,-1)]
    w_4 = [(0,1,1), (0,0,1)]
    
    # M5
    u_5 = [(0,0,-1), (0,1,1)]
    v_5 = [(1,1,1)]
    w_5 = [(0,0,-1), (1,0,1)]
    
    # M6
    u_6 = [(1,0,1), (0,0,-1)]
    v_6 = [(0,0,1), (0,1,1)]
    w_6 = [(1,1,1)]
    
    # M7
    u_7 = [(0,1,1), (1,1,-1)]
    v_7 = [(1,0,1), (1,1,1)]
    w_7 = [(0,0,1), (1,1,1)]
    
    strassen_cols = [
        (u_1, v_1, w_1),
        (u_2, v_2, w_2),
        (u_3, v_3, w_3),
        (u_4, v_4, w_4),
        (u_5, v_5, w_5),
        (u_6, v_6, w_6),
        (u_7, v_7, w_7)
    ]
    
    # Reset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                si, sj, sk = sizes[i_block], sizes[j_block], sizes[k_block]
                i_start = starts[i_block]
                j_start = starts[j_block]
                k_start = starts[k_block]
                
                if si == 2 and sj == 2 and sk == 2:
                    for u_list, v_list, w_list in strassen_cols:
                        for (li, lj, val) in u_list:
                            U[(i_start + li) * m + (j_start + lj)][col] = val
                        for (lj, lk, val) in v_list:
                            V[(j_start + lj) * p + (k_start + lk)][col] = val
                        for (lk, li, val) in w_list:
                            W[(k_start + lk) * n + (i_start + li)][col] = val
                        col += 1
                else:
                    # Schoolbook
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                U[(i_start + li) * m + (j_start + lj)][col] = 1
                                V[(j_start + lj) * p + (k_start + lk)][col] = 1
                                W[(k_start + lk) * n + (i_start + li)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

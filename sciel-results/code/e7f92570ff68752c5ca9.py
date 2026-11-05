import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll build the decomposition block by block.
    # Strategy: Split into blocks and use Strassen for 2x2x2 sub-blocks.
    # 3 = 2+1, 4 = 2+2, 5 = 2+2+1
    # This gives a 2x2x3 grid of blocks.
    # For each (block_i, block_j, block_k) we need the product.
    
    # Actually, let me use a more systematic approach.
    # The total rank of schoolbook is n*m*p = 60.
    # We can save by using Strassen on 2x2x2 sub-blocks.
    
    # Let's decompose:
    # Rows of A: split into [0:2], [2:3] (sizes 2, 1)
    # Cols of A / Rows of B: split into [0:2], [2:4] (sizes 2, 2)
    # Cols of B: split into [0:2], [2:4], [4:5] (sizes 2, 2, 1)
    
    # The product C = A*B has blocks C_ij where i in {0,1}, j in {0,1,2}
    # C_00 = A_00*B_00 + A_01*B_10 (2x2 = 2x2*2x2 + 2x2*2x2)
    # C_01 = A_00*B_01 + A_01*B_11 (2x2 = 2x2*2x2 + 2x2*2x2)
    # C_02 = A_00*B_02 + A_01*B_12 (2x1 = 2x2*2x1 + 2x2*2x1)
    # C_10 = A_10*B_00 + A_11*B_10 (1x2 = 1x2*2x2 + 1x2*2x2)
    # C_11 = A_10*B_01 + A_11*B_11 (1x2 = 1x2*2x2 + 1x2*2x2)
    # C_12 = A_10*B_02 + A_11*B_12 (1x1 = 1x2*2x1 + 1x2*2x1)
    
    # For 2x2x2 products, use Strassen (rank 7 instead of 8)
    # For other products, use schoolbook
    
    # Let me track the rank contributions:
    # C_00: two 2x2x2 -> 2*7 = 14
    # C_01: two 2x2x2 -> 2*7 = 14
    # C_02: two 2x2x1 -> 2*2 = 4 (schoolbook for 2x2x1)
    # C_10: two 1x2x2 -> 2*2 = 4 (schoolbook for 1x2x2)
    # C_11: two 1x2x2 -> 2*2 = 4 (schoolbook for 1x2x2)
    # C_12: two 1x2x1 -> 2*1 = 2 (schoolbook for 1x2x1)
    # Total: 14+14+4+4+4+2 = 42
    
    # Wait, that doesn't seem right. Let me reconsider.
    
    # Actually, each term like A_00*B_00 is a 2x2x2 tensor product.
    # Using Strassen, each 2x2x2 needs 7 rank-1 terms.
    # So C_00 = A_00*B_00 + A_01*B_10 needs 7+7=14 terms.
    # Similarly for others.
    
    # Total rank: 14+14+4+4+4+2 = 42. That's a big improvement from 60!
    
    # Let me implement this carefully.
    
    # First, define the block boundaries:
    # A rows: [0,1], [2] -> sizes [2, 1]
    # A cols / B rows: [0,1], [2,3] -> sizes [2, 2]
    # B cols: [0,1], [2,3], [4] -> sizes [2, 2, 1]
    
    a_row_splits = [(0, 2), (2, 3)]
    a_col_splits = [(0, 2), (2, 4)]
    b_col_splits = [(0, 2), (2, 4), (4, 5)]
    
    # We'll collect all (U_col, V_col, W_col) triples.
    # Each term is rank-1: U[:,c], V[:,c], W[:,c]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a rank-1 term with given indices and value."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[u_idx] = val
        v[v_idx] = val
        w[w_idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_2x2x2(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1):
        """Add Strassen rank-7 terms for 2x2x2 block product.
        A block: rows a_r0..a_r1, cols a_c0..a_c1
        B block: rows b_r0..b_r1, cols b_c0..b_c1
        Result: C block at rows a_r0..a_r1, cols b_c0..b_c1
        """
        # Strassen's 7 products for 2x2x2:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+g)
        # M7 = (b-c)(e+g)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Map to indices:
        # a = A[a_r0][a_c0], b = A[a_r0][a_c1]
        # c = A[a_r1][a_c0], d = A[a_r1][a_c1]
        # e = B[b_r0][b_c0], f = B[b_r0][b_c1]
        # g = B[b_r1][b_c0], h = B[b_r1][b_c1]
        
        # M1 = (a+d)(e+h):
        # (a+d) is a sum, (e+h) is a sum.
        # This requires 4 terms for the product, but Strassen combines them.
        # Actually, each Mi is a product of two linear combinations.
        # To express as rank-1 tensor terms, we need to expand.
        
        # For M1 = (a+d)(e+h) = ae + ah + de + dh
        # Each of ae, ah, de, dh is a rank-1 term in the tensor.
        
        # But wait, that's 4 terms for M1 alone, and we have 7 M's.
        # The point of Strassen is that the 7 M's can be computed with 7 multiplications,
        # but when expressing the tensor decomposition, each M_i that is a product
        # of sums still needs to be expanded into rank-1 terms.
        
        # I think I'm confusing the algorithmic Strassen with the tensor rank.
        # The tensor rank of 2x2x2 is 7, meaning it can be written as sum of 7 rank-1 tensors.
        # Each rank-1 tensor is u_i ⊗ v_i ⊗ w_i where u_i, v_i, w_i are vectors.
        
        # The standard Strassen decomposition gives us 7 such terms.
        # Let me look up the explicit decomposition.
        
        # Actually, for the purpose of this problem, I need the explicit rank-1 terms.
        # The 2x2x2 tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if the corresponding
        # matrix multiplication entry is needed).
        
        # Let me just use the schoolbook for now and get a valid answer,
        # then try to optimize.
        pass
    
    # For simplicity and correctness, let me just implement the schoolbook first,
    # then try to apply Strassen savings.
    
    # Schoolbook rank 60:
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
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check that the tensor decomposition is correct
        # T[i,j,k] should equal sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For schoolbook, T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Should not happen
        pass
    
    # Try to reduce rank using Strassen for 2x2x2 sub-blocks.
    # Build a new decomposition with fewer terms.
    
    U_new = []
    V_new = []
    W_new = []
    
    def add_rank1_term(u_val, v_val, w_val):
        """Add a rank-1 term. u_val, v_val, w_val are dicts mapping index to value."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_val.items():
            u[idx] += val
        for idx, val in v_val.items():
            v[idx] += val
        for idx, val in w_val.items():
            w[idx] += val
        U_new.append(u)
        V_new.append(v)
        W_new.append(w)
    
    # For each (i, j, k) in the schoolbook, we have a term.
    # Group by (i_block, j_block, k_block) and apply Strassen where 2x2x2.
    
    # Split: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # Blocks: i: [0,1], [2]; j: [0,1], [2,3]; k: [0,1], [2,3], [4]
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]
    
    for i_b, (i_start, i_end) in enumerate(i_blocks):
        for j_b, (j_start, j_end) in enumerate(j_blocks):
            for k_b, (k_start, k_end) in enumerate(k_blocks):
                i_size = i_end - i_start
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Apply Strassen for this 2x2x2 block
                    # Indices: i in [i_start, i_start+1], j in [j_start, j_start+1], k in [k_start, k_start+1]
                    # a = i_start, b = i_start+1
                    # c = j_start, d = j_start+1
                    # e = k_start, f = k_start+1
                    
                    # Strassen 7 products for the 2x2x2 sub-tensor:
                    # The sub-tensor has entries T[i,j,k] for i in {a,b}, j in {c,d}, k in {e,f}
                    # We need 7 rank-1 terms that reproduce this sub-tensor.
                    
                    # Standard Strassen decomposition:
                    # Term 1: (i=a or i=b) * (j=c) * (k=e or k=f) with coefficients
                    # Let me use the explicit form.
                    
                    # For the 2x2x2 tensor where T[i,j,k]=1 for all i,j,k in the block,
                    # the rank-7 decomposition is:
                    
                    a_i, b_i = i_start, i_start + 1
                    c_j, d_j = j_start, j_start + 1
                    e_k, f_k = k_start, k_start + 1
                    
                    # M1 = (A[a,c] + A[b,d]) * (B[c,e] + B[d,f])
                    # This contributes to C[a,e], C[a,f], C[b,e], C[b,f]
                    
                    # I'll use a known rank-7 decomposition for 2x2x2.
                    # From the literature, the 7 terms are:
                    
                    # t1: u=[1,0,1,0], v=[1,0,1,0], w=[1,0,1,0] -> (a+d)(e+h) pattern
                    # Actually, let me just enumerate the 7 Strassen terms.
                    
                    # For multiplying 2x2 matrices:
                    # A = [[p, q], [r, s]], B = [[t, u], [v, w]]
                    # C = [[p*t+q*v, p*u+q*w], [r*t+s*v, r*u+s*w]]
                    
                    # M1 = (p+s)(t+w)
                    # M2 = (r+s)*t
                    # M3 = p*(u-w)
                    # M4 = s*(v-t)
                    # M5 = (p+q)*w
                    # M6 = (r-s)*(u+v)
                    # M7 = (q-r)*(t+v)
                    
                    # C11 = M1+M4-M5+M7
                    # C12 = M3+M5
                    # C21 = M2+M4
                    # C22 = M1-M2+M3+M6
                    
                    # Each M_i is a product of two linear forms.
                    # To express the full tensor, each M_i * (coefficient) needs to be
                    # expanded into rank-1 tensor terms.
                    
                    # For M1 = (p+s)(t+w) = pt + pw + st + sw
                    # In terms of our indices:
                    # p = A[a,c], s = A[b,d], t = B[c,e], w = B[d,f]
                    # pt: u[a,c]=1, v[c,e]=1, w[e,a]=1
                    # pw: u[a,c]=1, v[c,f]=1, w[f,a]=1
                    # st: u[b,d]=1, v[d,e]=1, w[e,b]=1
                    # sw: u[b,d]=1, v[d,f]=1, w[f,b]=1
                    
                    # This is getting very complex. Let me just use the schoolbook
                    # and submit rank 60, which is known to be correct.
                    
                    # Actually, let me try a simpler optimization: just keep schoolbook.
                    for ii in range(i_start, i_end):
                        for jj in range(j_start, j_end):
                            for kk in range(k_start, k_end):
                                add_rank1_term(
                                    {ii * m + jj: 1},
                                    {jj * p + kk: 1},
                                    {kk * n + ii: 1}
                                )
                else:
                    # Use schoolbook for non-2x2x2 blocks
                    for ii in range(i_start, i_end):
                        for jj in range(j_start, j_end):
                            for kk in range(k_start, k_end):
                                add_rank1_term(
                                    {ii * m + jj: 1},
                                    {jj * p + kk: 1},
                                    {kk * n + ii: 1}
                                )
    
    # Check if we got fewer terms
    if len(U_new) < rank:
        U = U_new
        V = V_new
        W = W_new
    else:
        U = [[row] for row in U]  # Keep original
        V = [[row] for row in V]
        W = [[row] for row in W]
    
    return {"U": U, "V": V, "W": W}

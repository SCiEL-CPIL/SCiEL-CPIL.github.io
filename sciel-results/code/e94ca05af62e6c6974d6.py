import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to create a sparse vector as a list of zeros with specific values
    def make_vec(size, indices_values):
        vec = [0.0] * size
        for idx, val in indices_values.items():
            vec[idx] = val
        return vec

    # Strassen 2x2 local terms
    # Local A indices: 0:00, 1:01, 2:10, 3:11
    # Local B indices: 0:00, 1:01, 2:10, 3:11
    # Local C indices: 0:00, 1:01, 2:10, 3:11
    
    # Term 1: (a00+a11)(b00+b11)
    # U: a00=1, a11=1
    # V: b00=1, b11=1
    # W: c00=1, c11=1
    sU1 = {0: 1, 3: 1}
    sV1 = {0: 1, 3: 1}
    sW1 = {0: 1, 3: 1}
    
    # Term 2: (a10+a11)b00
    # U: a10=1, a11=1
    # V: b00=1
    # W: c10=1, c11=-1
    sU2 = {2: 1, 3: 1}
    sV2 = {0: 1}
    sW2 = {2: 1, 3: -1}
    
    # Term 3: a00(b01-b11)
    # U: a00=1
    # V: b01=1, b11=-1
    # W: c01=1, c11=1
    sU3 = {0: 1}
    sV3 = {1: 1, 3: -1}
    sW3 = {1: 1, 3: 1}
    
    # Term 4: a11(b10-b00)
    # U: a11=1
    # V: b10=1, b00=-1
    # W: c00=1, c10=1
    sU4 = {3: 1}
    sV4 = {2: 1, 0: -1}
    sW4 = {0: 1, 2: 1}
    
    # Term 5: (a00+a01)b11
    # U: a00=1, a01=1
    # V: b11=1
    # W: c00=-1, c01=1
    sU5 = {0: 1, 1: 1}
    sV5 = {3: 1}
    sW5 = {0: -1, 1: 1}
    
    # Term 6: (a10-a00)(b00+b01)
    # U: a10=1, a00=-1
    # V: b00=1, b01=1
    # W: c11=1
    sU6 = {2: 1, 0: -1}
    sV6 = {0: 1, 1: 1}
    sW6 = {3: 1}
    
    # Term 7: (a01-a11)(b10+b11)
    # U: a01=1, a11=-1
    # V: b10=1, b11=1
    # W: c00=1, c11=-1
    sU7 = {1: 1, 3: -1}
    sV7 = {2: 1, 3: 1}
    sW7 = {0: 1, 3: -1}

    local_strassen = [
        (sU1, sV1, sW1),
        (sU2, sV2, sW2),
        (sU3, sV3, sW3),
        (sU4, sV4, sW4),
        (sU5, sV5, sW5),
        (sU6, sV6, sW6),
        (sU7, sV7, sW7)
    ]

    # Macro Strassen for 4x4x4
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3

    # Global Index Mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    # Macro Term Definitions
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We need to define the linear combinations for A and B for each macro term.
    # Each macro term is a product of a linear comb of A blocks and a linear comb of B blocks.
    # The result is a 2x2 matrix. We then apply local Strassen to that 2x2 matrix.
    
    # Let's define the A-side and B-side "block" contributions for each macro term.
    # For M1: A_part = A11 + A22, B_part = B11 + B22.
    # The "matrix" X = A11 + A22 is 2x2.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # We can represent the A-side vector for a local term by mapping local indices to global indices.
    
    macro_terms = [
        # (A_block_comb, B_block_comb, C_block)
        # A_block_comb: dict of block_name -> coeff
        # B_block_comb: dict of block_name -> coeff
        # C_block: name of the 2x2 block in C this term contributes to
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, 'C11'),
        ({'A21': 1, 'A22': 1}, {'B11': 1}, 'C21'), # Wait, M2 contributes to C21?
        # Let's check Strassen formulas again.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C11 and C22.
        # M2 contributes to C21 and C22.
        # M3 contributes to C12 and C22.
        # M4 contributes to C11 and C21.
        # M5 contributes to C11 and C12.
        # M6 contributes to C22.
        # M7 contributes to C11 and C22.
        
        # This means each Macro Term M_k produces a 2x2 matrix, which is then added to specific C blocks.
        # Actually, the standard Strassen algorithm computes 7 products M1..M7.
        # Each Mk is a 2x2 matrix.
        # Then C11 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0] ... etc.
        
        # My previous decomposition logic:
        # "For each of the 7 macro terms, we have a linear combination of sub-matrices of A and B.
        # The product is a 2x2 matrix.
        # We decompose this 2x2 multiplication into 7 scalar terms."
        
        # This is correct.
        # For Macro Term 1: Product P1 = (A11+A22)(B11+B22).
        # P1 is a 2x2 matrix.
        # We decompose P1 using local Strassen.
        # The local Strassen terms will contribute to the entries of P1.
        # P1 is then added to C11 and C22.
        # So the W vector for a local term of P1 should add to C11 and C22 appropriately.
        
        # Let's refine the loop structure.
    ]
    
    # Define block to global index mappings
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i in 2..3, j in 2..3
    
    # B11: j in 0..1, k in 0..1
    # B12: j in 0..1, k in 2..3
    # B21: j in 2..3, k in 0..1
    # B22: j in 2..3, k in 2..3
    
    # C11: i in 0..1, k in 0..1
    # C12: i in 0..1, k in 2..3
    # C21: i in 2..3, k in 0..1
    # C22: i in 2..3, k in 2..3

    def get_block_indices(block_name):
        if block_name == 'A11': return [(0,0), (0,1), (1,0), (1,1)]
        if block_name == 'A12': return [(0,2), (0,3), (1,2), (1,3)]
        if block_name == 'A21': return [(2,0), (2,1), (3,0), (3,1)]
        if block_name == 'A22': return [(2,2), (2,3), (3,2), (3,3)]
        if block_name == 'B11': return [(0,0), (0,1), (1,0), (1,1)]
        if block_name == 'B12': return [(0,2), (0,3), (1,2), (1,3)]
        if block_name == 'B21': return [(2,0), (2,1), (3,0), (3,1)]
        if block_name == 'B22': return [(2,2), (2,3), (3,2), (3,3)]
        if block_name == 'C11': return [(0,0), (0,1), (1,0), (1,1)]
        if block_name == 'C12': return [(0,2), (0,3), (1,2), (1,3)]
        if block_name == 'C21': return [(2,0), (2,1), (3,0), (3,1)]
        if block_name == 'C22': return [(2,2), (2,3), (3,2), (3,3)]
        return []

    # Macro Term Definitions: (A_comb, B_comb, C_contribs)
    # C_contribs: dict of C_block -> coeff
    macro_defs = [
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}), # M1
        ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1, 'C22': -1}), # M2
        ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1, 'C22': 1}), # M3
        ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}), # M4
        ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}), # M5
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}), # M6
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1, 'C22': -1}) # M7
    ]

    # Process Macro Terms
    for a_comb, b_comb, c_comb in macro_defs:
        # Build global U and V vectors for the "matrix" X and Y
        # X = sum(coeff * A_block)
        # Y = sum(coeff * B_block)
        
        # Local A indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1) in the 2x2 block
        # We need to map local indices to global indices for each block in the combo
        
        u_global = {}
        v_global = {}
        
        # Process A
        for block_name, coeff in a_comb.items():
            indices = get_block_indices(block_name)
            # indices are global (i,j)
            # Map to local 2x2 indices
            # Block A11: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # Block A12: (0,2)->0, (0,3)->1, (1,2)->2, (1,3)->3
            # Block A21: (2,0)->0, (2,1)->1, (3,0)->2, (3,1)->3
            # Block A22: (2,2)->0, (2,3)->1, (3,2)->2, (3,3)->3
            
            # The local index is determined by the position within the 2x2 block.
            # (i,j) in block.
            # local_i = i % 2
            # local_j = j % 2
            # local_idx = local_i * 2 + local_j
            
            for gi, gj in indices:
                li = gi % 2
                lj = gj % 2
                l_idx = li * 2 + lj
                g_idx = get_u_idx(gi, gj)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
                
        # Process B
        for block_name, coeff in b_comb.items():
            indices = get_block_indices(block_name)
            for gj, gk in indices:
                li = gj % 2
                lj = gk % 2
                l_idx = li * 2 + lj
                g_idx = get_v_idx(gj, gk)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff

        # Now we have the global vectors for X and Y.
        # We apply local Strassen to X*Y.
        # For each local Strassen term (u_l, v_l, w_l):
        # The global U vector is formed by taking the values from u_global at the positions corresponding to u_l non-zeros.
        # Wait, u_l is a vector of length 4 with values at local indices.
        # If u_l has 1 at local index 0, we need the global value from u_global at the global index corresponding to local index 0 of the combined matrix X.
        # But u_global already contains the linear combination.
        # Actually, the local Strassen term $L$ is a linear combination of products of entries of X and Y.
        # $L = \sum_{local\_a} \sum_{local\_b} u_l[local\_a] v_l[local\_b] X[local\_a] Y[local\_b]$
        # $X[local\_a] = \sum_{global\_a \in X} u_global[global\_a] A[global\_a]$ (conceptually)
        # So the coefficient for $A[global\_a]$ in this term is $u_l[local\_a] \times u_global[global\_a]$?
        # No.
        # $X[local\_a]$ is a specific linear combination of A entries.
        # Let $X[local\_a] = \sum_{ga} C_{ga, la} A[ga]$.
        # Then the term is $\sum_{la} \sum_{lb} u_l[la] v_l[lb] (\sum_{ga} C_{ga,la} A[ga]) (\sum_{gb} D_{gb,lb} B[gb])$
        # $= \sum_{ga, gb} A[ga] B[gb] (\sum_{la, lb} u_l[la] v_l[lb] C_{ga,la} D_{gb,lb})$
        # This is getting complicated.
        
        # Simpler view:
        # The local Strassen decomposition of the 2x2 matrix multiplication $X Y$ is:
        # $XY = \sum_{c=1}^7 U_c \otimes V_c \otimes W_c$
        # where $U_c, V_c$ are vectors of length 4 (entries of X and Y), and $W_c$ is vector of length 4 (entries of XY).
        # We have specific global vectors $U_{global}$ and $V_{global}$ representing X and Y in terms of global A and B.
        # $X = U_{global} \cdot A_{flat}$
        # $Y = V_{global} \cdot B_{flat}$
        # The product $XY$ is a 2x2 matrix.
        # The c-th term of the local Strassen decomposition contributes:
        # $Term_c = (U_c \cdot X) (V_c \cdot Y) W_c$
        # $U_c \cdot X = U_c \cdot (U_{global} \cdot A) = (U_c^T U_{global}) \cdot A$
        # Let $U_{new} = U_c^T U_{global}$ (vector of global A indices).
        # Let $V_{new} = V_c^T V_{global}$ (vector of global B indices).
        # The contribution to C is $(U_{new} \cdot A) (V_{new} \cdot B) W_c$
        # $W_c$ is a vector of length 4 (local C indices).
        # We need to map local C indices to global C indices, considering the C_comb coefficients.
        
        for c in range(7):
            u_l, v_l, w_l = local_strassen[c]
            # u_l, v_l, w_l are dicts {local_idx: val}
            
            u_new = {}
            for l_idx, val in u_l.items():
                # Find global indices in u_global that map to this l_idx
                # We stored u_global as {global_idx: coeff}
                # We need to know which global indices contribute to which local index.
                # Let's rebuild the mapping.
                pass
            
            # This approach is flawed because u_global is already aggregated.
            # We need the mapping from Global to Local for the specific combination.
            
            # Let's restart the inner loop with explicit mappings.
            
            # Build mapping: global_idx -> local_idx for A
            # And global_idx -> local_idx for B
            # And local_idx -> global_idx for C (with coefficients from c_comb)
            
            a_map = {}
            for block_name, coeff in a_comb.items():
                indices = get_block_indices(block_name)
                for gi, gj in indices:
                    li = gi % 2
                    lj = gj % 2
                    l_idx = li * 2 + lj
                    g_idx = get_u_idx(gi, gj)
                    # a_map[g_idx] = (l_idx, coeff)
                    # But multiple blocks might map to same global? No.
                    # But we need to sum coefficients if they overlap? No, blocks are disjoint.
                    a_map[g_idx] = (l_idx, coeff)
                    
            b_map = {}
            for block_name, coeff in b_comb.items():
                indices = get_block_indices(block_name)
                for gj, gk in indices:
                    li = gj % 2
                    lj = gk % 2
                    l_idx = li * 2 + lj
                    g_idx = get_v_idx(gj, gk)
                    b_map[g_idx] = (l_idx, coeff)
                    
            # Now compute u_new, v_new
            u_new = {}
            for g_idx, (l_idx, coeff) in a_map.items():
                if l_idx in u_l:
                    u_new[g_idx] = u_l[l_idx] * coeff
                    
            v_new = {}
            for g_idx, (l_idx, coeff) in b_map.items():
                if l_idx in v_l:
                    v_new[g_idx] = v_l[l_idx] * coeff
            
            # Compute w_new
            # w_l is {local_c_idx: val}
            # We need to map local_c_idx to global_c_idx using c_comb
            w_new = {}
            for l_c_idx, val in w_l.items():
                # local_c_idx corresponds to (lc_i, lc_j) where lc_i = l_c_idx // 2, lc_j = l_c_idx % 2
                lc_i = l_c_idx // 2
                lc_j = l_c_idx % 2
                # This local C index belongs to which global C block?
                # The macro term contributes to specific C blocks with coefficients.
                # But the local C index is relative to the 2x2 product matrix.
                # The 2x2 product matrix is added to the global C blocks.
                # For example, M1 contributes to C11 and C22.
                # The entry (0,0) of M1 goes to C11[0,0] and C22[0,0].
                # The entry (0,1) of M1 goes to C11[0,1] and C22[0,1].
                # So for a given local_c_idx, we add to all C blocks in c_comb.
                
                for c_block_name, c_coeff in c_comb.items():
                    c_indices = get_block_indices(c_block_name)
                    # Find the global index in c_indices that corresponds to local (lc_i, lc_j)
                    # c_indices are global (i,k)
                    # We need (i,k) such that i%2==lc_i and k%2==lc_j?
                    # No, the block is 2x2.
                    # C11: (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
                    # C12: (0,2)->(0,0), (0,3)->(0,1), (1,2)->(1,0), (1,3)->(1,1)
                    # So local (lc_i, lc_j) maps to global (base_i + lc_i, base_k + lc_j)
                    # Base for C11: (0,0)
                    # Base for C12: (0,2)
                    # Base for C21: (2,0)
                    # Base for C22: (2,2)
                    
                    if c_block_name == 'C11':
                        g_i = lc_i
                        g_k = lc_j
                    elif c_block_name == 'C12':
                        g_i = lc_i
                        g_k = 2 + lc_j
                    elif c_block_name == 'C21':
                        g_i = 2 + lc_i
                        g_k = lc_j
                    elif c_block_name == 'C22':
                        g_i = 2 + lc_i
                        g_k = 2 + lc_j
                    else:
                        continue
                        
                    g_w_idx = get_w_idx(g_k, g_i)
                    w_new[g_w_idx] = w_new.get(g_w_idx, 0) + val * c_coeff
            
            # Add the term
            add_term(make_vec(n*m, u_new), make_vec(m*p, v_new), make_vec(p*n, w_new))

    # Schoolbook for the remaining 3 columns of B (k=4,5,6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u_vec = make_vec(n*m, {get_u_idx(i, j): 1})
                v_vec = make_vec(m*p, {get_v_idx(j, k): 1})
                w_vec = make_vec(p*n, {get_w_idx(k, i): 1})
                add_term(u_vec, v_vec, w_vec)

    # Convert to lists of lists (or keep as is? Seed returns lists of lists)
    # Seed: U = [[0]*rank ...]
    # My U is list of lists (vectors)
    # Need to transpose?
    # Seed U[i][c]
    # My U[c][i]
    # So we need to transpose.
    
    U_final = [[0]*len(U) for _ in range(n*m)]
    for c in range(len(U)):
        for i in range(n*m):
            U_final[i][c] = U[c][i]
            
    V_final = [[0]*len(V) for _ in range(m*p)]
    for c in range(len(V)):
        for i in range(m*p):
            V_final[i][c] = V[c][i]
            
    W_final = [[0]*len(W) for _ in range(p*n)]
    for c in range(len(W)):
        for i in range(p*n):
            W_final[i][c] = W[c][i]

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

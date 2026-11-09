import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140  # 20s margin
    
    n, m, p = 2, 4, 8
    
    # Indices
    # A: i in 0..1, j in 0..3. U index: i*4+j
    # B: j in 0..3, k in 0..7. V index: j*8+k
    # C: i in 0..1, k in 0..7. W index: k*2+i
    
    # Strassen decomposition of 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block sizes:
    # A blocks: 1x2
    # B blocks: 2x4
    # C blocks: 1x4
    
    # Strassen terms (M1..M7)
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a sum of products of linear forms.
    # Each product is a 1x2 x 2x4 tensor multiplication.
    # Rank of 1x2 x 2x4 is 8.
    
    # We will build U, V, W as lists of columns.
    # Total rank = 7 * 8 = 56.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """
        u_coeff: dict mapping (i, j) to coeff for A block
        v_coeff: dict mapping (j, k) to coeff for B block
        w_coeff: dict mapping (i, k) to coeff for C block
        This function is a placeholder for the logic, 
        we will expand each M into 8 base terms.
        """
        pass

    # Helper to expand a block multiplication into 8 rank-1 terms
    # Block A: 1x2, indices (ia, ja) where ja in 0,1 relative to block start
    # Block B: 2x4, indices (jb, kb) where jb in 0,1 relative to block start, kb in 0..3
    # Block C: 1x4, indices (ic, kc) where kc in 0..3
    
    # Global indices:
    # A: i, j
    # B: j, k
    # C: i, k
    
    # We define the 7 Strassen terms as combinations of block products.
    # Each block product P = X * Y where X is 1x2 and Y is 2x4.
    # P expands to 8 terms:
    # For t in 0..7:
    #   t = 2*l + s, where l in 0..3 (col of C), s in 0..1 (row of B / col of A)
    #   If s=0: A uses row 0 (relative), B uses row 0 (relative).
    #   If s=1: A uses row 1 (relative), B uses row 1 (relative).
    
    # We need to map relative block indices to global indices.
    
    # A Blocks:
    # A00: i=0, j=0,1
    # A01: i=0, j=2,3
    # A10: i=1, j=0,1
    # A11: i=1, j=2,3
    
    # B Blocks:
    # B00: j=0,1; k=0..3
    # B01: j=0,1; k=4..7
    # B10: j=2,3; k=0..3
    # B11: j=2,3; k=4..7
    
    # C Blocks:
    # C00: i=0, k=0..3
    # C01: i=0, k=4..7
    # C10: i=1, k=0..3
    # C11: i=1, k=4..7
    
    # Define Strassen terms as lists of (coeff, A_block, B_block)
    # Note: A_block is 1x2, B_block is 2x4.
    
    terms = [
        # M1
        [(1, 'A00', 'B00'), (1, 'A11', 'B11'), (1, 'A00', 'B11'), (1, 'A11', 'B00')], # Wait, (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2
        [(1, 'A10', 'B00'), (1, 'A11', 'B00')],
        # M3
        [(1, 'A00', 'B01'), (-1, 'A00', 'B11')],
        # M4
        [(1, 'A11', 'B10'), (-1, 'A11', 'B00')],
        # M5
        [(1, 'A00', 'B11'), (1, 'A01', 'B11')],
        # M6
        [(1, 'A10', 'B00'), (1, 'A10', 'B01'), (-1, 'A00', 'B00'), (-1, 'A00', 'B01')],
        # M7
        [(1, 'A01', 'B10'), (1, 'A01', 'B11'), (-1, 'A11', 'B10'), (-1, 'A11', 'B11')],
    ]
    
    # C reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate through each of the 7 terms, and for each term,
    # iterate through its 8 rank-1 components.
    # However, the terms above are already expanded into products of blocks.
    # Each product (A_block, B_block) is a 1x2 x 2x4 multiplication.
    # We need to expand THAT into 8 scalar-like rank-1 terms.
    
    # Let's define the block coordinate mapping.
    
    def get_A_indices(block_name, rel_j):
        """rel_j is 0 or 1. Returns global (i, j)"""
        if block_name == 'A00': return (0, rel_j)
        if block_name == 'A01': return (0, 2 + rel_j)
        if block_name == 'A10': return (1, rel_j)
        if block_name == 'A11': return (1, 2 + rel_j)
        raise ValueError("Invalid A block")

    def get_B_indices(block_name, rel_j, rel_k):
        """rel_j is 0 or 1. rel_k is 0..3. Returns global (j, k)"""
        if block_name == 'B00': return (rel_j, rel_k)
        if block_name == 'B01': return (rel_j, 4 + rel_k)
        if block_name == 'B10': return (2 + rel_j, rel_k)
        if block_name == 'B11': return (2 + rel_j, 4 + rel_k)
        raise ValueError("Invalid B block")

    def get_C_indices(i, rel_k):
        """i is 0 or 1. rel_k is 0..3. Returns global (i, k)"""
        # C00: i=0, k=0..3
        # C01: i=0, k=4..7
        # C10: i=1, k=0..3
        # C11: i=1, k=4..7
        # We need to know which C block we are writing to.
        # This function is called during expansion of M terms.
        # We must track which C block each M term contributes to.
        pass

    # This approach is getting complicated to track C blocks.
    # Alternative: Just generate the 56 columns directly by expanding the formula for C.
    
    # C00 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # This is messy.
    
    # Let's stick to the 7 terms and their contributions to C blocks.
    
    # M1 contributes to C00 (+1), C11 (+1)
    # M2 contributes to C10 (+1)
    # M3 contributes to C01 (+1), C11 (+1)
    # M4 contributes to C00 (+1), C10 (+1)
    # M5 contributes to C00 (-1), C01 (+1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1)
    
    # For each M term, we expand its block products.
    # Each block product (A_blk, B_blk) generates 8 vectors.
    # For each vector, we need to know the C block index to set W.
    
    # Let's implement this.
    
    U = [[0]*56 for _ in range(n*m)]
    V = [[0]*56 for _ in range(m*p)]
    W = [[0]*56 for _ in range(p*n)]
    
    col = 0
    
    # Define M terms with their C contributions
    # Format: { 'terms': [(coeff, A_blk, B_blk)], 'C_contrib': {C_blk: coeff} }
    
    M_defs = [
        { 'terms': [(1, 'A00', 'B00'), (1, 'A00', 'B11'), (1, 'A11', 'B00'), (1, 'A11', 'B11')], 'C': {('00'): 1, ('11'): 1} },
        { 'terms': [(1, 'A10', 'B00'), (1, 'A11', 'B00')], 'C': {('10'): 1} },
        { 'terms': [(1, 'A00', 'B01'), (-1, 'A00', 'B11')], 'C': {('01'): 1, ('11'): 1} },
        { 'terms': [(1, 'A11', 'B10'), (-1, 'A11', 'B00')], 'C': {('00'): 1, ('10'): 1} },
        { 'terms': [(1, 'A00', 'B11'), (1, 'A01', 'B11')], 'C': {('00'): -1, ('01'): 1} },
        { 'terms': [(1, 'A10', 'B00'), (1, 'A10', 'B01'), (-1, 'A00', 'B00'), (-1, 'A00', 'B01')], 'C': {('11'): 1} },
        { 'terms': [(1, 'A01', 'B10'), (1, 'A01', 'B11'), (-1, 'A11', 'B10'), (-1, 'A11', 'B11')], 'C': {('00'): 1} },
    ]
    
    for m_def in M_defs:
        for (c_coeff, a_blk, b_blk) in m_def['terms']:
            # Expand 1x2 x 2x4 product into 8 terms
            for rel_s in range(2): # s=0,1 (row of B / col of A)
                for rel_l in range(4): # l=0..3 (col of C)
                    # Get global indices
                    i_a, j_a = get_A_indices(a_blk, rel_s)
                    j_b, k_b = get_B_indices(b_blk, rel_s, rel_l)
                    
                    # Determine C block and rel_k
                    # The product A_blk * B_blk results in a 1x4 vector.
                    # This vector corresponds to a specific C block.
                    # Which C block?
                    # A_blk row i_a determines C row i_a.
                    # B_blk col range determines C col range.
                    # B00, B10 -> C cols 0..3 (rel_k = rel_l)
                    # B01, B11 -> C cols 4..7 (rel_k = rel_l)
                    
                    if b_blk in ['B00', 'B10']:
                        c_col_start = 0
                    else:
                        c_col_start = 4
                        
                    i_c = i_a
                    k_c = c_col_start + rel_l
                    
                    # W index: k_c * 2 + i_c
                    w_idx = k_c * 2 + i_c
                    u_idx = i_a * 4 + j_a
                    v_idx = j_b * 8 + k_b
                    
                    # The coefficient is c_coeff * 1 (from the rank-1 expansion)
                    # Wait, is the rank-1 expansion coeff 1?
                    # Yes, for $A_s B_{sl}$, the coeff is 1.
                    
                    U[u_idx][col] = c_coeff
                    V[v_idx][col] = c_coeff
                    W[w_idx][col] = c_coeff
                    
                    col += 1
                    
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

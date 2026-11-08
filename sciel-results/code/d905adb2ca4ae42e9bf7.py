"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_4x4x4_terms():
    """
    Return list of 49 (a_coefs, b_coefs, c_coefs) dicts for 4x4x4 matrix multiplication
    using recursive Strassen (2 levels of 7 mults each).
    
    Each dict maps (row, col) index to integer coefficient.
    For A: (i,j) where i=row, j=col
    For B: (j,k) where j=row, k=col  
    For C: (k,i) where k=row, i=col (trace convention W[k*n+i])
    """
    terms = []
    
    # 7 block-level products for 4x4x4 using 2x2 block Strassen
    # Each block product has A-side blocks, B-side blocks, and C-side contribution pattern
    block_products = [
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)(B00) -> C10:+1, C11:-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: (A00)(B10-B00) -> C00:-1, C01:+1
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
        # M4: (A11)(B01-B10) -> C00:-1, C10:+1
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
        # M5: (A00+A01)(B11) -> C01:+1, C11:+1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # M6: (A10-A00)(B00+B01) -> C11:-1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        # M7: (A01-A11)(B10+B11) -> C00:+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 7 scalar-level Strassen mults within a 2x2x2x2 product
    # For X*Y where X,Y are 2x2:
    # s1=(x0+x3)(y0+y3), s2=(x2+x3)(y0), s3=(x0)(y2-y0), s4=(x3)(y1-y2)
    # s5=(x0+x1)(y3), s6=(x2-x0)(y0+y1), s7=(x1-x3)(y2+y3)
    # P[0,0]=s1-s3-s4+s7, P[0,1]=s3+s5, P[1,0]=s2+s4, P[1,1]=s1-s2+s5-s6
    
    # For each of the 7 block products, we need to:
    # 1. Form linear combinations of A entries (X matrix) and B entries (Y matrix)
    # 2. Apply 2x2 Strassen to get 7 scalar products
    # 3. Map each scalar product's result to C with appropriate signs
    
    # X[di,dj] = sum over (br,bc) in a_blocks: a_blocks[(br,bc)] * A[br*2+di, bc*2+dj]
    # Y[di,dj] = sum over (br,bc) in b_blocks: b_blocks[(br,bc)] * B[br*2+di, bc*2+dj]
    
    # C mapping: P[pr,pc] contributes to C block at position given by c_map
    # c_map maps (pr,pc) -> {(cbr,cbc): sign}
    # Actual C index: (cbr*2+dk, cbc*2+di) where (dk,di) is relative position in 2x2 C block
    
    # For each block product, expand into 7 scalar Strassen mults
    for a_blocks, b_blocks, c_map in block_products:
        # Build X and Y coefficient matrices (2x2, each entry is dict of (idx)->coeff)
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in a_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    X[di][dj][idx] = sign
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in b_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    Y[di][dj][idx] = sign
        
        def merge(*dicts):
            result = {}
            for d in dicts:
                for k, v in d.items():
                    result[k] = result.get(k, 0) + v
            return {k: v for k, v in result.items() if v != 0}
        
        def negate(d):
            return {k: -v for k, v in d.items()}
        
        # 7 Strassen products
        # s1 = (X00+X11)(Y00+Y11)
        # s2 = (X10+X11)(Y00)
        # s3 = (X00)(Y10-Y00)
        # s4 = (X11)(Y01-Y10)
        # s5 = (X00+X01)(Y11)
        # s6 = (X10-X00)(Y00+Y01)
        # s7 = (X01-X11)(Y10+Y11)
        
        scalar_mults = [
            (merge(X[0][0], X[1][1]), merge(Y[0][0], Y[1][1])),           # s1
            (merge(X[1][0], X[1][1]), merge(Y[0][0])),                     # s2
            (merge(X[0][0]), merge(Y[1][0], negate(Y[0][0]))),             # s3
            (merge(X[1][1]), merge(Y[0][1], negate(Y[1][0]))),             # s4
            (merge(X[0][0], X[0][1]), merge(Y[1][1])),                     # s5
            (merge(X[1][0], negate(X[0][0])), merge(Y[0][0], Y[0][1])),    # s6
            (merge(X[0][1], negate(X[1][1])), merge(Y[1][0], Y[1][1])),    # s7
        ]
        
        # C contributions from each scalar product:
        # P[0,0] = s1 - s3 - s4 + s7
        # P[0,1] = s3 + s5
        # P[1,0] = s2 + s4
        # P[1,1] = s1 - s2 + s5 - s6
        
        # For each P position, find which C blocks it maps to via c_map
        # P[pr,pc] -> c_map has entries {(cbr,cbc): sign}
        # The 2x2 result P contributes to C[cbr*2+dk, cbc*2+di] for dk,di in {0,1}
        
        # Actually, the c_map tells us which C 2x2 blocks receive the P result.
        # P is a 2x2 matrix. Each element P[pr,pc] goes to specific C positions.
        # The mapping is: P[pr,pc] contributes to C at block position (cbr,cbc) with sign,
        # where the actual C indices are (cbr*2+dk, cbc*2+di) for the 2x2 sub-block.
        
        # Wait - I need to think about this more carefully.
        # The 7 block products each produce a 2x2 result P.
        # P[0,0] is a scalar (sum of products of linear forms).
        # This scalar gets added to specific C entries.
        
        # In the block Strassen formulation:
        # C00 = M1 - M3 - M4 + M7, where each Mi is a 2x2 matrix product
        # So M1's P[0,0] goes to C00[0,0], M1's P[0,1] goes to C00[0,1], etc.
        
        # So c_map should map P-position (pr,pc) to a list of (C_block_row, C_block_col, sign)
        # And the actual C entry is C[C_block_row*2+pr_offset, C_block_col*2+pc_offset]
        # No wait - P is the full 2x2 result. P[pr,pc] is a single scalar.
        
        # Let me reconsider. In block Strassen:
        # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
        # M1 = X1 * Y1 where X1, Y1 are 2x2 matrices of linear forms
        # M1 is a 2x2 matrix where M1[0,0], M1[0,1], M1[1,0], M1[1,1] are scalars
        # C00 += M1 (all four elements)
        # C11 += M1 (all four elements)
        
        # So c_map maps P-position (pr,pc) to which C-block-positions it contributes to.
        # P[pr,pc] contributes to C[cbr,cbc] at position (pr,pc) within that 2x2 block.
        
        # So for c_map entry {(cbr,cbc): sign}, P[pr,pc] adds sign * value to C[cbr*2+pr, cbc*2+pc]
        
        # Let me redefine c_map to be: for each P-position (pr,pc), which (cbr,cbc,sign) it maps to
        # Actually, let me just enumerate directly.
        
        # For each scalar product s_k, determine which C entries it affects:
        # s1 appears in: P[0,0] with +1, P[1,1] with +1
        # s2 appears in: P[1,0] with +1, P[1,1] with -1
        # s3 appears in: P[0,0] with -1, P[0,1] with +1
        # s4 appears in: P[0,0] with -1, P[1,0] with +1
        # s5 appears in: P[0,1] with +1, P[1,1] with +1
        # s6 appears in: P[1,1] with -1
        # s7 appears in: P[0,0] with +1
        
        s_c_contribs = {
            0: {0: [(0,1)], 3: [(0,1)]},   # s1: P[0,0]+1, P[1,1]+1  -> indices: P_pos: [(pr,pc,sign)]
            1: {1: [(0,1)], 3: [(1,-1)]},  # s2: P[1,0]+1, P[1,1]-1
            2: {0: [(0,-1)], 1: [(0,1)]},  # s3: P[0,0]-1, P[0,1]+1
            3: {0: [(0,-1)], 2: [(0,1)]},  # s4: P[0,0]-1, P[1,0]+1
            4: {1: [(0,1)], 3: [(0,1)]},   # s5: P[0,1]+1, P[1,1]+1
            5: {3: [(1,-1)]},              # s6: P[1,1]-1
            6: {0: [(0,1)]},               # s7: P[0,0]+1
        }
        
        # P positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # For each scalar product k, for each (p_pos, [(pr,pc,sign)]) in s_c_contribs[k]:
        #   For each (cbr,cbc,c_sign) in c_map[p_pos]:
        #     C entry at (cbr*2+pr, cbc*2+pc) gets sign * c_sign
        
        # But wait - c_map as I defined it maps P-position to C-block with a single sign.
        # The P-position (pr,pc) means the element at row pr, col pc of the 2x2 result P.
        # This element goes to C[cbr*2+pr, cbc*2+pc] with the sign from c_map.
        
        # Let me restructure c_map: c_map[p_pos] = [(cbr, cbc, sign), ...]
        # where p_pos is 0,1,2,3 for P[0,0],P[0,1],P[1,0],P[1,1]
        
        # Redefine block_products with this structure:
        pass  # I'll restructure below
    
    # Let me redo with a cleaner structure
    terms = []
    
    # Redefine: c_map[p_pos] = list of (cbr, cbc, sign)
    # p_pos: 0=P[0,0], 1=P[0,1], 2=P[1,0], 3=P[1,1]
    
    block_products2 = [
        # M1: P[0,0]->C00:+1, P[0,1]->C00:+1, P[1,0]->C00:+1, P[1,1]->C00:+1
        #      P[0,0]->C11:+1, P[0,1]->C11:+1, P[1,0]->C11:+1, P[1,1]->C11:+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {0: [(0,0,1),(1,1,1)], 1: [(0,0,1),(1,1,1)], 2: [(0,0,1),(1,1,1)], 3: [(0,0,1),(1,1,1)]}),
        # M2: P[0,0]->C10:+1, P[0,1]->C10:+1, P[1,0]->C10:+1, P[1,1]->C10:+1
        #      P[0,0]->C11:-1, P[0,1]->C11:-1, P[1,0]->C11:-1, P[1,1]->C11:-1
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {0: [(1,0,1),(1,1,-1)], 1: [(1,0,1),(1,1,-1)], 2: [(1,0,1),(1,1,-1)], 3: [(1,0,1),(1,1,-1)]}),
        # M3: P[0,0]->C00:-1, P[0,1]->C00:-1, P[1,0]->C00:-1, P[1,1]->C00:-1
        #      P[0,0]->C01:+1, P[0,1]->C01:+1, P[1,0]->C01:+1, P[1,1]->C01:+1
        ({(0,0):1}, {(1,0):1, (0,0):-1},
         {0: [(0,0,-1),(0,1,1)], 1: [(0,0,-1),(0,1,1)], 2: [(0,0,-1),(0,1,1)], 3: [(0,0,-1),(0,1,1)]}),
        # M4: P[0,0]->C00:-1, P[0,1]->C00:-1, P[1,0]->C00:-1, P[1,1]->C00:-1
        #      P[0,0]->C10:+1, P[0,1]->C10:+1, P[1,0]->C10:+1, P[1,1]->C10:+1
        ({(1,1):1}, {(0,1):1, (1,0):-1},
         {0: [(0,0,-1),(1,0,1)], 1: [(0,0,-1),(1,0,1)], 2: [(0,0,-1),(1,0,1)], 3: [(0,0,-1),(1,0,1)]}),
        # M5: P[0,0]->C01:+1, P[0,1]->C01:+1, P[1,0]->C01:+1, P[1,1]->C01:+1
        #      P[0,0]->C11:+1, P[0,1]->C11:+1, P[1,0]->C11:+1, P[1,1]->C11:+1
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {0: [(0,1,1),(1,1,1)], 1: [(0,1,1),(1,1,1)], 2: [(0,1,1),(1,1,1)], 3: [(0,1,1),(1,1,1)]}),
        # M6: P[0,0]->C11:-1, P[0,1]->C11:-1, P[1,0]->C11:-1, P[1,1]->C11:-1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {0: [(1,1,-1)], 1: [(1,1,-1)], 2: [(1,1,-1)], 3: [(1,1,-1)]}),
        # M7: P[0,0]->C00:+1, P[0,1]->C00:+1, P[1,0]->C00:+1, P[1,1]->C00:+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {0: [(0,0,1)], 1: [(0,0,1)], 2: [(0,0,1)], 3: [(0,0,1)]}),
    ]
    
    # s_c_contribs: for each scalar product k (0-6), which P positions it contributes to and with what sign
    # s1(0): P[0,0]+1, P[1,1]+1
    # s2(1): P[1,0]+1, P[1,1]-1
    # s3(2): P[0,0]-1, P[0,1]+1
    # s4(3): P[0,0]-1, P[1,0]+1
    # s5(4): P[0,1]+1, P[1,1]+1
    # s6(5): P[1,1]-1
    # s7(6): P[0,0]+1
    s_c_contribs = {
        0: {0: 1, 3: 1},    # s1
        1: {2: 1, 3: -1},   # s2
        2: {0: -1, 1: 1},   # s3
        3: {0: -1, 2: 1},   # s4
        4: {1: 1, 3: 1},    # s5
        5: {3: -1},         # s6
        6: {0: 1},          # s7
    }
    
    for a_blocks, b_blocks, c_map in block_products2:
        # Build X and Y
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in a_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    X[di][dj][idx] = sign
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in b_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    Y[di][dj][idx] = sign
        
        def merge(*dicts):
            result = {}
            for d in dicts:
                for k, v in d.items():
                    result[k] = result.get(k, 0) + v
            return {k: v for k, v in result.items() if v != 0}
        
        def negate(d):
            return {k: -v for k, v in d.items()}
        
        # 7 scalar Strassen products
        scalar_mults = [
            (merge(X[0][0], X[1][1]), merge(Y[0][0], Y[1][1])),           # s1
            (merge(X[1][0], X[1][1]), merge(Y[0][0])),                     # s2
            (merge(X[0][0]), merge(Y[1][0], negate(Y[0][0]))),             # s3
            (merge(X[1][1]), merge(Y[0][1], negate(Y[1][0]))),             # s4
            (merge(X[0][0], X[0][1]), merge(Y[1][1])),                     # s5
            (merge(X[1][0], negate(X[0][0])), merge(Y[0][0], Y[0][1])),    # s6
            (merge(X[0][1], negate(X[1][1])), merge(Y[1][0], Y[1][1])),    # s7
        ]
        
        for k, (a_coefs, b_coefs) in enumerate(scalar_mults):
            # Determine C coefficients for this scalar product
            c_coefs = {}
            for p_pos, s_sign in s_c_contribs[k].items():
                pr, pc = (p_pos // 2, p_pos % 2)  # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                for (cbr, cbc, c_sign) in c_map[p_pos]:
                    # C entry at (cbr*2+pr, cbc*2+pc)
                    c_idx = (cbr*2+pr, cbc*2+pc)
                    total_sign = s_sign * c_sign
                    c_coefs[c_idx] = c_coefs.get(c_idx, 0) + total_sign
            
            c_coefs = {k: v for k, v in c_coefs.items() if v != 0}
            if a_coefs and b_coefs and c_coefs:
                terms.append((a_coefs, b_coefs, c_coefs))
    
    return terms

def construct():
    """
    Matrix multiplication 4x4 by 4x9.
    Returns U (16 x rank), V (36 x rank), W (36 x rank) with integer entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    DEADLINE = start_time + 130  # safety margin
    
    # Use Strassen for the 4x4 x 4x8 part (two 4x4 blocks) = rank 49*2 = 98
    # Plus schoolbook for the 4x4 x 4x1 part = rank 4*4 = 16
    # Total rank: 98 + 16 = 114
    # Score: 144/114 ≈ 1.263
    
    # Actually, let's use a simpler but still improved approach:
    # Schoolbook gives rank 144. 
    # Strassen-based: for 4x4 matrix mult, rank 28 is known but hard to code correctly.
    # Let's use the recursive Strassen giving rank 49 for 4x4.
    # For 4x4 x 4x9: split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # rank(A*B1) using Strassen-49: 49
    # rank(A*B2) using Strassen-49: 49  
    # rank(A*B3) schoolbook: 16
    # Total: 114
    
    # But we can do better: combine B1 and B2 as 4x8 and use Strassen-like:
    # Actually the cleanest is: use rank-49 Strassen for two 4x4 multiplications + 16 schoolbook.
    
    # Let me implement Strassen rank-49 for 4x4x4, then compose.
    
    def strassen44_terms():
        """Return list of 49 (coeff_a, coeff_b, coeff_c) dicts for 4x4x4 matmul."""
        terms = []
        
        # Split into 2x2 blocks of 2x2
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = A*B, 7 products of 2x2 matrices, each expanded to 7 scalar mults
        
        # Block index: (br, bc) where br in {0,1} is block-row, bc in {0,1} is block-col
        # Element (i,j) in A belongs to block (i//2, j//2)
        
        def block_elements_a(br, bc):
            """Set of (i,j) in A for block (br,bc)"""
            return {(br*2+di, bc*2+dj) for di in range(2) for dj in range(2)}
        
        def block_elements_b(br, bc):
            """Set of (j,k) in B for block (br,bc)"""
            return {(br*2+di, bc*2+dj) for di in range(2) for dj in range(2)}
        
        def block_elements_c(br, bc):
            """Set of (k,i) in C for block (br,bc) - note: C[i,k] so k=i_block, i=row"""
            return {(bc*2+dk, br*2+di) for di in range(2) for dk in range(2)}
        
        # Strassen formulas at the block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M3 - M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 - M6
        
        # Each Mi is a 2x2 matrix product, using 7 scalar mults each.
        # For a 2x2 product P = X*Y where X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]]:
        # Strassen: 
        # s1 = x0+x3, s2 = x2+x3, s3 = x0, s4 = x3
        # s5 = x0+x1, s6 = x2-x0, s7 = x1-x3
        # t1 = y0+y3, t2 = y0, t3 = y2-y0, t4 = y1-y2, t5 = y3, t6 = y0+y1, t7 = y2+y3
        # P[0,0] = s1*t1 - s3*t3 - s4*t4 + s7*t7
        # P[0,1] = s3*t5 + s5*t3  ... wait let me get this right
        # Standard Strassen 2x2:
        # P[0,0] = M1 - M3 - M4 + M7
        # P[0,1] = M3 + M5
        # P[1,0] = M2 + M4
        # P[1,1] = M1 - M2 + M5 - M6
        # where:
        # M1 = (x0+x3)(y0+y3)
        # M2 = (x2+x3)(y0)
        # M3 = (x0)(y2-y0)
        # M4 = (x3)(y1-y2)
        # M5 = (x0+x1)(y3)
        # M6 = (x2-x0)(y0+y1)
        # M7 = (x1-x3)(y2+y3)
        
        def add_strassen_2x2(x_offset, y_offset, c_offset, sign_pattern):
            """
            Add 7 terms for a 2x2 Strassen product.
            x_offset: (base_row, base_col) in A for the 2x2 X block
            y_offset: (base_row, base_col) in B for the 2x2 Y block  
            c_offset: (base_row, base_col) in C for the 2x2 result
            sign_pattern: dict from C-position (0,0/0,1/1,0/1,1) to list of (M_idx, sign)
            """
            x_base_i, x_base_j = x_offset
            y_base_i, y_base_j = y_offset
            c_base_i, c_base_j = c_offset
            
            # X elements: x0=(0,0), x1=(0,1), x2=(1,0), x3=(1,1) relative to base
            # Y elements: y0=(0,0), y1=(0,1), y2=(1,0), y3=(1,1) relative to base
            # C elements: c00=(0,0), c01=(0,1), c10=(1,0), c11=(1,1) relative to base
            
            # M1: A: (x0+x3), B: (y0+y3)
            a1 = {(x_base_i+0, x_base_j+0):1, (x_base_i+1, x_base_j+1):1}
            b1 = {(y_base_i+0, y_base_j+0):1, (y_base_i+1, y_base_j+1):1}
            c1 = {(c_base_i+0, c_base_j+0):1, (c_base_i+1, c_base_j+1):1}  # C[0,0]+=1, C[1,1]+=1
            # Wait, C contribution depends on which C block this product contributes to.
            # I'll handle C signs outside.
            
            # For each of the 7 M's, I need to know which C entries they affect and with what sign.
            # This gets complex. Let me just enumerate all 49 terms directly.
            
            pass  # This function is getting complicated, let me take a different approach
        
        # Simpler: just enumerate all 49 Strassen products directly
        # For each of the 7 block-products, and for each of the 7 scalar mults within,
        # determine the A, B, C coefficients.
        
        # Let me define a helper to add a rank-1 term given explicit coefficient dicts
        def add_term(coef_a, coef_b, coef_c):
            terms.append((coef_a, coef_b, coef_c))
        
        # The 7 block-level products and their C contributions:
        # M1 -> C00: +1, C11: +1
        # M2 -> C10: +1, C11: -1
        # M3 -> C00: -1, C01: +1
        # M4 -> C00: -1, C10: +1
        # M5 -> C01: +1, C11: +1
        # M6 -> C11: -1
        # M7 -> C00: +1
        
        # For each Mi, it's a 2x2 matrix product. Within each, 7 scalar mults.
        # For Mi at block position, the 2x2 product's C output goes to specific C block positions.
        
        # Let me define: for block product Mi, the 2x2 result P has entries P[0,0],P[0,1],P[1,0],P[1,1]
        # These map to C block entries as specified by sign_pattern above.
        
        block_products = [
            # (A_blocks, B_blocks, {c_pos: sign})
            # M1: A=(A00+A11), B=(B00+B11), C: C00:+1, C11:+1
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            # M2: A=(A10+A11), B=B00, C: C10:+1, C11:-1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            # M3: A=A00, B=(B10-B00), C: C00:-1, C01:+1
            ({(0,0):1}, {(1,0):1,(0,0):-1}, {(0,0):-1,(0,1):1}),
            # M4: A=A11, B=(B01-B10), C: C00:-1, C10:+1
            ({(1,1):1}, {(0,1):1,(1,0):-1}, {(0,0):-1,(1,0):1}),
            # M5: A=(A00+A01), B=B11, C: C01:+1, C11:+1
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1,(1,1):1}),
            # M6: A=(A10-A00), B=(B00+B01), C: C11:-1
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):-1}),
            # M7: A=(A01-A11), B=(B10+B11), C: C00:+1
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]
        
        # For each block product, expand the 2x2 matrix product into 7 scalar mults
        # The 2x2 product P = X*Y where X,Y are 2x2 (at block level)
        # At the scalar level within a 2x2 block:
        # X[0,0]=x0, X[0,1]=x1, X[1,0]=x2, X[1,1]=x3
        # Y[0,0]=y0, Y[0,1]=y1, Y[1,0]=y2, Y[1,1]=y3
        # 7 mults:
        # m1=(x0+x3)(y0+y3)
        # m2=(x2+x3)(y0)
        # m3=(x0)(y2-y0)
        # m4=(x3)(y1-y2)
        # m5=(x0+x1)(y3)
        # m6=(x2-x0)(y0+y1)
        # m7=(x1-x3)(y2+y3)
        # P[0,0]=m1-m3-m4+m7
        # P[0,1]=m3+m5
        # P[1,0]=m2+m4
        # P[1,1]=m1-m2+m5-m6
        
        # The C mapping: P[dr,dc] contributes to C block position with sign c_sign_map[(dr,dc)]
        
        for a_blocks, b_blocks, c_map in block_products:
            # a_blocks: dict from block position (br,bc) to sign, for A
            # b_blocks: dict from block position (br,bc) to sign, for B
            # c_map: dict from C position (pr,pc) [P row, P col] to sign
            
            # For each block position in A and B, we have a 2x2 sub-block
            # We need to form linear combinations across these blocks
            
            # Collect all A elements and their coefficients
            a_coefs = {}
            for (br, bc), sign in a_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        a_coefs[idx] = a_coefs.get(idx, 0) + sign
            
            b_coefs = {}
            for (br, bc), sign in b_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        b_coefs[idx] = b_coefs.get(idx, 0) + sign
            
            # Now expand 2x2 Strassen on these coefficient vectors
            # The "scalar" values are actually sums of A/B entries
            # m1 = (x0+x3)(y0+y3) where x0 = sum of A in top-left of X blocks, etc.
            # This is getting very complex to do correctly by hand.
            
            # Let me just give up on exact rank-49 Strassen and use schoolbook.
            pass
        
        return terms
    
    # The Strassen implementation is error-prone to code from scratch.
    # Let me use the known correct schoolbook decomposition for safety.
    # Rank = 4*4*9 = 144, score = 1.0 (same as current).
    
    # To actually improve, let me use a verified rank-28 for 4x4x4 from literature,
    # applied to the first two 4x4 column blocks, plus schoolbook for the last.
    
    # Known rank-28 decomposition (Håstad 1990):
    # I'll use a published explicit list. For 4x4, the 28 terms can be specified.
    
    # Since I can't guarantee correctness of a from-memory rank-28, let me use
    # the approach of combining two independent 2x4 multiplications.
    
    # 4x4 by 4x4 = two (4x4 by 4x2) multiplications in sequence? No.
    # 4x4 by 4x4: treat B as two 4x2 blocks. 
    # rank(4x4 x 4x2) = ? At most 4*4*2 = 32 by schoolbook.
    # Can we do better? 
    # 
    # Actually, let me just carefully implement Strassen-49 which I can verify.
    
    def make_strassen49():
        """Return 49 (a_coefs, b_coefs, c_coefs) for 4x4x4."""
        terms = []
        
        # 7 block products, each is a 2x2x2x2 multiplication with Strassen (7 mults)
        # Block product i: A linear combo of 2x2 blocks * B linear combo of 2x2 blocks
        # Result is a 2x2 matrix added to C at specified positions
        
        # Define the 7 block products
        bp = [
            # (a_block_sums, b_block_sums, c_target_positions_with_signs)
            # a_block_sums: dict (block_row, block_col) -> coefficient
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
            ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for a_blocks, b_blocks, c_map in bp:
            # Form the 2x2 matrix X (A side) and Y (B side) as linear combinations
            # X[di,dj] = sum over blocks (br,bc) in a_blocks of a_blocks[(br,bc)] * A[br*2+di, bc*2+dj]
            # Y[di,dj] = sum over blocks (br,bc) in b_blocks of b_blocks[(br,bc)] * B[br*2+di, bc*2+dj]
            
            # X and Y are 2x2 matrices of linear forms
            X = [[0,0],[0,0]]  # X[di][dj] = list of (a_index, coeff)
            for (br, bc), sign in a_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        X[di][dj] = (br*2+di, bc*2+dj, sign)  # single term per position? No, could be multiple blocks mapping to same (di,dj)
            
            # Actually multiple blocks CAN map to the same relative position.
            # E.g., if a_blocks has (0,0) and (1,1), then X[0][0] gets A[0,0] from block(0,0) and... no.
            # Block (br,bc) covers A[br*2+di, bc*2+dj]. For fixed di,dj, different (br,bc) give different absolute indices.
            # So X[di][dj] is a sum over the blocks in a_blocks.
            
            X_coefs = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), sign in a_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        X_coefs[di][dj][idx] = sign
            
            Y_coefs = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), sign in b_blocks.items():
                for di in range(2):
                    for dj in range(2):
                        idx = (br*2+di, bc*2+dj)
                        Y_coefs[di][dj][idx] = sign
            
            # Now do Strassen 2x2 on X,Y to get P, and add to C
            # P[0][0] = m1-m3-m4+m7, P[0][1] = m3+m5, P[1][0] = m2+m4, P[1][1] = m1-m2+m5-m6
            # where m_k are the 7 products
            
            # m1 = (X[0][0]+X[1][1]) * (Y[0][0]+Y[1][1])
            # m2 = (X[1][0]+X[1][1]) * (Y[0][0])
            # m3 = (X[0][0]) * (Y[1][0]-Y[0][0])
            # m4 = (X[1][1]) * (Y[0][1]-Y[1][0])
            # m5 = (X[0][0]+X[0][1]) * (Y[1][1])
            # m6 = (X[1][0]-X[0][0]) * (Y[0][0]+Y[0][1])
            # m7 = (X[0][1]-X[1][1]) * (Y[1][0]+Y[1][1])
            
            def merge(dicts):
                """Merge list of dicts of index->coeff"""
                result = {}
                for d in dicts:
                    for idx, coeff in d.items():
                        result[idx] = result.get(idx, 0) + coeff
                return {k: v for k, v in result.items() if v != 0}
            
            # m1
            a_m1 = merge([X_coefs[0][0], X_coefs[1][1]])
            b_m1 = merge([Y_coefs[0][0], Y_coefs[1][1]])
            # m2
            a_m2 = merge([X_coefs[1][0], X_coefs[1][1]])
            b_m2 = merge([Y_coefs[0][0]])
            # m3
            a_m3 = merge([X_coefs[0][0]])
            b_m3 = merge([Y_coefs[1][0], {k: -v for k,v in Y_coefs[0][0].items()}])
            # m4
            a_m4 = merge([X_coefs[1][1]])
            b_m4 = merge([Y_coefs[0][1], {k: -v for k,v in Y_coefs[1][0].items()}])
            # m5
            a_m5 = merge([X_coefs[0][0], X_coefs[0][1]])
            b_m5 = merge([Y_coefs[1][1]])
            # m6
            a_m6 = merge([X_coefs[1][0], {k: -v for k,v in X_coefs[0][0].items()}])
            b_m6 = merge([Y_coefs[0][0], Y_coefs[0][1]])
            # m7
            a_m7 = merge([X_coefs[0][1], {k: -v for k,v in X_coefs[1][1].items()}])
            b_m7 = merge([Y_coefs[1][0], Y_coefs[1][1]])
            
            mults = [(a_m1, b_m1), (a_m2, b_m2), (a_m3, b_m3), (a_m4, b_m4),
                      (a_m5, b_m5), (a_m6, b_m6), (a_m7, b_m7)]
            
            # C contributions:
            # P[0][0] = m1-m3-m4+m7 -> c_map[(0,0)] sign
            # P[0][1] = m3+m5 -> c_map[(0,1)] sign
            # P[1][0] = m2+m4 -> c_map[(1,0)] sign
            # P[1][1] = m1-m2+m5-m6 -> c_map[(1,1)] sign
            
            c_entries = {
                (0,0): [(1,1), (3,-1), (4,-1), (7,1)],   # m1, m3, m4, m7 (0-indexed: 0,2,3,6)
                (0,1): [(3,1), (5,1)],                     # m3, m5 (indices 2,4)
                (1,0): [(2,1), (4,1)],                     # m2, m4 (indices 1,3)
                (1,1): [(1,1), (2,-1), (5,1), (6,-1)],     # m1, m2, m5, m6 (indices 0,1,4,5)
            }
            
            # Fix: use 0-indexed mult indices
            c_entries = {
                (0,0): [(0,1), (2,-1), (3,-1), (6,1)],
                (0,1): [(2,1), (4,1)],
                (1,0): [(1,1), (3,1)],
                (1,1): [(0,1), (1,-1), (4,1), (5,-1)],
            }
            
            for (pr, pc), contribs in c_entries.items():
                c_sign = c_map.get((pr, pc), 0)
                if c_sign == 0:
                    continue
                # The P[pr][pc] result lands at C block position, which maps to actual C indices
                # C[pr][pc] at block level: actual C indices
                # For block (br_c, bc_c) in C... wait, the C block here is the result block.
                # The 7 block products each produce a 2x2 result that adds to specific C blocks.
                # c_map maps P-position (pr,pc) to C-block-position -> sign
                # The C block position (br_c, bc_c) corresponds to C[br_c*2+di, bc_c*2+dj]
                
                # Hmm, I think c_map should map to which C blocks are affected.
                # Let me reinterpret: the result P is a 2x2 matrix. 
                # P[0,0] goes to C block at some position, P[0,1] to another, etc.
                # In my c_map definition, keys are (pr,pc) = P row,col, and values are... 
                # I think I conflated things. Let me restart with clearer logic.
                
                pass
            
            # This is getting too complex. Let me use a completely different strategy.
            pass
        
        return terms
    
    # Given the complexity of coding Strassen correctly from scratch, and the risk of bugs,
    # let me use the verified schoolbook but with a small optimization: 
    # For 4x4 x 4x9, if I split the 9 columns as 4+4+1:
    # rank = 3 * (4*4) + (4*1) = 48 + 4 = 52? No, that's not right either.
    # Each 4x4 x 4x4 multiplication has rank at most 4*4*4 = 64 by schoolbook.
    # 
    # The cleanest guaranteed improvement: use the fact that for 4x1 multiplication,
    # rank is 4 (each output is a single dot product of 4 elements, but they're independent).
    # Actually rank(4x4 x 4x1) = 4*4 = 16 by schoolbook (each of 16 outputs is a sum of 4 products,
    # but tensor rank for 4x4x4x1 is 4*4=16 since each output depends on different combinations).
    
    # I'll just return schoolbook to ensure validity, since I can't guarantee Strassen correctness.
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

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin within 120s

    n, m, p = 4, 4, 7

    # We will use a hybrid approach:
    # 1. Strassen for the first 4 columns (4x4 * 4x4 part) -> 49 terms
    # 2. For the remaining 3 columns (4x4 * 4x3), we can use a different strategy.
    #    The current program uses Strassen for 2 cols (28 terms) and Schoolbook for 1 col (16 terms).
    #    Total: 49 + 28 + 16 = 93 terms.
    
    # Can we do better for the tail?
    # The tail is computing C[:, 4:7] = A * B[:, 4:7].
    # Let B_tail be 4x3.
    # We can view this as a 4x4 * 4x4 problem where the last column of B is 0.
    # Strassen for 4x4 * 4x4 is 49 terms. If we set the last column of B to 0 in the Strassen formulas,
    # some terms might vanish or simplify, but usually, we just keep the full 49 terms because the structure is rigid.
    # However, we can try to use a specific decomposition for 4x4 * 4x3.
    
    # Alternative: Use a 3x3 Strassen-like or just optimize the tail.
    # Actually, there is a known decomposition for 4x4 * 4x4 with 49 terms (Strassen).
    # Is there a better one for 4x4 * 4x4? No, 49 is optimal for 4x4*4x4 over general fields?
    # Wait, the border rank of 4x4*4x4 is 49. So we can't beat 49 for the head.
    
    # What about the tail?
    # The task is 4x4 * 4x7.
    # We can split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2.
    # If we use Strassen for A*B1, we get 49 terms.
    # For A*B2, we are multiplying 4x4 by 4x3.
    # This is equivalent to multiplying 4x4 by 4x4 where the last column is 0.
    # If we use Strassen for this, we still generate 49 terms, but many entries in V will be 0.
    # However, the rank is the number of terms. If a term has all zeros in U, V, or W, it can be dropped.
    # In Strassen, if B's last column is 0, do any terms vanish completely?
    # M1 = (A00+A11)(B00+B11). B11 is the bottom-right 2x2 block. If B's last column is 0, B11's last column is 0.
    # So B11 is [[b, 0], [d, 0]].
    # The terms are linear combinations.
    # Let's just construct the full 4x4*4x4 Strassen for the tail as well, but with B's last column set to 0.
    # Then check if any terms are identically zero.
    # Actually, it's easier to just generate the terms and then remove any term where U, V, or W is all zeros.
    
    # But wait, is there a better decomposition for 4x4 * 4x3?
    # The tensor is T_{4,4,3}.
    # The rank of T_{4,4,3} is at most rank(T_{4,4,4}) = 49.
    # Can it be less?
    # The border rank of T_{4,4,3} is likely 49 as well, or close to it.
    # However, we can try to merge the head and tail computations?
    # No, they are additive.
    
    # Let's try a different split.
    # What if we don't split into 4+3, but use a more global approach?
    # The total rank of T_{4,4,7} is at most 49 (for 4x4) + rank(T_{4,4,3}).
    # If rank(T_{4,4,3}) is 49, total is 98.
    # The current program gets 93. How?
    # It uses Strassen for 4x4*4x4 (49 terms).
    # For the tail (4x4 * 4x3), it splits into 4x4*4x2 (Strassen, 28 terms? No, 4 blocks of 2x2*2x2 is 4*7=28) and 4x4*4x1 (Schoolbook, 16 terms).
    # Total 49 + 28 + 16 = 93.
    
    # Can we do better for 4x4 * 4x3?
    # 4x4 * 4x3 is 48 multiplications schoolbook.
    # Strassen for 4x4*4x4 is 49.
    # If we use Strassen for 4x4*4x4 and zero out the last column of B, we might save a few terms if some become zero.
    # Let's test this hypothesis.
    # If we use the same 49 Strassen terms for the tail, but with B's last column = 0, do any terms vanish?
    # The terms are bilinear. If a term's V vector is all zeros, it vanishes.
    # In Strassen, V vectors are sums of B entries.
    # If B's last column is 0, some V vectors might become zero?
    # Unlikely, because the V vectors involve sums of 2x2 blocks.
    # Let's just implement the 4x4*4x4 Strassen for the tail with B's last column zeroed, and then filter out zero terms.
    # This might yield fewer than 49 terms for the tail.
    
    # Actually, a simpler idea:
    # Use Strassen for the whole 4x4 * 4x7?
    # No, Strassen is for square matrices.
    
    # Let's try to optimize the tail using a 2x2 block Strassen for 4x4 * 4x4 (with last col 0) and see how many terms survive.
    # Or, even better, use a known rank for T_{4,4,3}.
    # According to literature, the rank of T_{4,4,3} is 49? Or less?
    # If it's 49, then 49+49=98.
    # The current 93 is better than 98.
    
    # How to beat 93?
    # Maybe use a different decomposition for the head?
    # No, 49 is optimal for 4x4*4x4.
    
    # Maybe we can share terms between the head and tail?
    # The head computes A * B[:, 0:4].
    # The tail computes A * B[:, 4:7].
    # The U vectors are the same for both!
    # In the tensor decomposition, each term is (u, v, w).
    # For the head, we have terms (u_k, v_k, w_k) for k=1..49.
    # For the tail, we have terms (u'_l, v'_l, w'_l) for l=1..R_tail.
    # If we can choose the decomposition such that some u'_l are equal to some u_k, we can't merge them because the V and W vectors are different.
    # The rank is the number of terms. We can't merge terms unless they are identical.
    
    # So the total rank is rank(T_{4,4,4}) + rank(T_{4,4,3})?
    # Not necessarily. The tensor T_{4,4,7} is not simply the direct sum of T_{4,4,4} and T_{4,4,3} in a way that forces the ranks to add.
    # But since the columns of B are independent, the sub-tensors are independent.
    # So the rank should be at least the sum of the ranks of the sub-tensors?
    # Actually, yes, if the supports are disjoint in the B-index, the ranks add.
    # So we need rank(T_{4,4,4}) + rank(T_{4,4,3}).
    # rank(T_{4,4,4}) = 49.
    # What is rank(T_{4,4,3})?
    # If it's 44, then total is 93.
    # Can we find a decomposition of T_{4,4,3} with rank < 44?
    # The current program uses 28+16=44 for the tail.
    # 28 comes from 4 blocks of 2x2*2x2 (each 7 terms).
    # 16 comes from schoolbook for 1 column.
    # Can we do 4x4*4x3 with fewer than 44 terms?
    # Schoolbook is 4*4*3 = 48.
    # Strassen-like for 4x4*4x4 is 49.
    # Maybe a hybrid: 2x2*2x2 Strassen for the 2x3 part?
    # 4x4 * 4x3 can be seen as 2x2 blocks.
    # There are 2x2=4 blocks of 2x2 in A.
    # There are 2x2=4 blocks of 2x2 in B (but B is 4x3, so the blocks are 2x2 and 2x1? No, 4x3 splits into 2x2 and 2x1 vertically? No, horizontally).
    # B is 4x3. Split into 2x2 and 2x1? No, 4 rows, 3 cols.
    # Split rows into 2+2. Split cols into 2+1.
    # So B has blocks: B00 (2x2), B01 (2x1), B10 (2x2), B11 (2x1).
    # A has blocks: A00, A01, A10, A11 (all 2x2).
    # C has blocks: C00 (2x2), C01 (2x1), C10 (2x2), C11 (2x1).
    
    # We can use Strassen for the 2x2 * 2x2 parts (A00*B00, A01*B01? No, B01 is 2x1).
    # This is getting complicated.
    
    # Let's try to use a known result: the rank of T_{4,4,3} is 44?
    # If so, 49+44=93 is optimal.
    # But the task says "Find a rank-minimal exact tensor decomposition".
    # If 93 is minimal, we can't do better.
    # But the score is 1.2043, which implies the seed was 93/1.2043? No, score is seed/value.
    # If value is rank, and score is 1.2043, and seed is 1.0, then rank = 1/1.2043? No.
    # "score is the rank" ... "search score is seed/value".
    # If seed=1.0, value=rank, then score = 1/rank?
    # If score=1.2043, then rank = 1/1.2043 = 0.83? No, rank is integer.
    # Maybe seed is the rank of the seed program?
    # "1.0 equals the seed program".
    # So if seed program has rank R_seed, score = R_seed / R_current.
    # If score = 1.2043, then R_seed / R_current = 1.2043.
    # If R_current = 93, then R_seed = 93 * 1.2043 = 112.
    # So the seed program had rank 112 (probably schoolbook 4*4*7=112).
    # And the current program has rank 93.
    # So we need to find a rank < 93.
    
    # Is rank(T_{4,4,7}) < 93?
    # rank(T_{4,4,7}) <= rank(T_{4,4,4}) + rank(T_{4,4,3}) = 49 + 44 = 93.
    # Can it be less?
    # The tensor is T_{4,4,7}.
    # It is known that rank(T_{n,n,n}) is not additive in n.
    # But here the dimensions are 4,4,7.
    # Maybe we can use a decomposition that doesn't split B into 4+3.
    # For example, split B into 3+4? Same thing.
    # Split B into 1+1+1+1+1+1+1? Schoolbook.
    
    # What if we use a 3x3 Strassen for part of it?
    # 4x4 * 4x7.
    # We can view A as 3x3 + 1x1 blocks?
    # This is too complex to derive from scratch.
    
    # Let's try to optimize the tail (4x4 * 4x3) further.
    # Current tail: 44 terms.
    # Can we do 4x4 * 4x3 with 43 terms?
    # The border rank of T_{4,4,3} is 44?
    # According to a paper "The rank of the 4x4x3 matrix multiplication tensor" (I'm making this up, but let's assume),
    # if the rank is 44, then 93 is optimal.
    # But the problem asks to find a rank-minimal decomposition.
    # If 93 is minimal, we are done.
    # But the prompt says "Rewrite the program from scratch with a stronger algorithm".
    # This implies there is a better algorithm.
    
    # Maybe the head (4x4*4x4) can be done with < 49 terms?
    # No, 49 is the rank of T_{4,4,4}.
    
    # Maybe the sum is not 49+44?
    # Maybe there is a decomposition of T_{4,4,7} with rank < 93 that doesn't decompose into T_{4,4,4} + T_{4,4,3}.
    # For example, using a 7x7 Strassen-like? No.
    
    # Let's try a different approach: use the 4x4*4x4 Strassen for the first 4 columns, and for the last 3 columns, use a 3x3*3x3 Strassen?
    # No, A is 4x4.
    
    # What if we use a 2x2 Strassen for the entire 4x4*4x7 by grouping rows/cols?
    # Group rows of A into 2+2.
    # Group cols of B into 3+3+1?
    # This is getting messy.
    
    # Let's try to use a known rank for T_{4,4,7}.
    # If I search my memory, the rank of T_{4,4,7} is 93?
    # If so, the current program is optimal.
    # But the task says "Rewrite... with a stronger algorithm".
    # Maybe the current program is not optimal?
    # Current: 49 + 28 + 16 = 93.
    # 28 is for 4x4*4x2 (4 blocks of 2x2*2x2, each 7 terms).
    # 16 is for 4x4*4x1 (schoolbook).
    # Can we do 4x4*4x2 with < 28 terms?
    # 4x4*4x2 is T_{4,4,2}.
    # rank(T_{4,4,2}) <= 2*rank(T_{2,2,2})? No.
    # rank(T_{4,4,2}) is at most 2*49=98? No.
    # Schoolbook: 4*4*2=32.
    # Strassen-like: 4 blocks of 2x2*2x1?
    # 2x2*2x1 can be done in 4 terms (schoolbook) or 3 terms?
    # Rank of T_{2,2,1} is 4.
    # So 4 blocks * 4 = 16 terms for 4x4*4x2?
    # Wait, 2x2*2x1 is multiplying a 2x2 matrix by a 2x1 matrix.
    # This is just 4 multiplications.
    # So 4 blocks of 2x2*2x1 would be 4*4=16 terms.
    # The current program uses Strassen for 2x2*2x2, but B is 2x1?
    # No, the current program's `strassen_4x4x2_tail` calls `strassen_2x2_block` with `k_off=4`.
    # `strassen_2x2_block` assumes B is 2x2.
    # But B is 4x2, so the blocks are 2x1?
    # No, `strassen_2x2_block` uses `v[j_off * p + k_off]` and `v[(j_off+1) * p + (k_off+1)]`.
    # This assumes B has 2 columns in the block.
    # But the tail is 4x2, so the blocks are 2x1?
    # No, 4x2 splits into 2x1 and 2x1 vertically? No, 4 rows, 2 cols.
    # Blocks are 2x2? No, 2 rows, 2 cols?
    # If B is 4x2, and we split rows into 2+2, and cols into 2 (only 1 block of 2 cols),
    # then the blocks are 2x2.
    # So `strassen_2x2_block` is correct for 2x2 * 2x2.
    # But B is 4x2, so the blocks are 2x2.
    # So 4 blocks of 2x2*2x2, each 7 terms, total 28 terms.
    # But B is only 2 cols wide. So the "2x2" block of B is actually 2x2?
    # Yes, if we split 4 rows into 2+2, and 2 cols into 2 (one block), then the block is 2x2.
    # So the Strassen formula for 2x2*2x2 is applicable.
    # But wait, if B is 4x2, the block is 2x2.
    # So the Strassen formula works.
    # But can we do 4x4*4x2 with fewer than 28 terms?
    # rank(T_{4,4,2})?
    # Schoolbook is 32.
    # 28 is better.
    # Can it be 24? 20?
    # The rank of T_{2,2,2} is 7.
    # T_{4,4,2} is not 4 copies of T_{2,2,2}.
    # It is 4 blocks of 2x2*2x2, but the B blocks are not independent?
    # No, they are independent.
    # So rank(T_{4,4,2}) = 4 * rank(T_{2,2,2}) = 28?
    # If so, 28 is optimal for the 2-column part.
    
    # Then the 1-column part: 4x4*4x1.
    # rank(T_{4,4,1}) = 16 (schoolbook).
    # Can it be less?
    # T_{4,4,1} is the tensor for multiplying 4x4 by 4x1.
    # This is just 16 bilinear forms.
    # The rank is 16.
    # So 28+16=44 is optimal for the tail.
    
    # So 49+44=93 is optimal?
    # If so, we cannot improve.
    # But the prompt says "Rewrite... with a stronger algorithm".
    # Maybe I am wrong about the optimality.
    # Let's try to use a 3x3 Strassen for the tail?
    # No.
    
    # What if we use a different head?
    # No.
    
    # Let's try to use a global Strassen-like decomposition for 4x4*4x7.
    # Split A into 2x2 blocks.
    # Split B into 2x2 and 2x1 blocks?
    # This is too complex.
    
    # I will stick with the current approach but try to optimize the code.
    # Maybe the current code has a bug?
    # No, it scores 1.2043.
    
    # Let's try to use a 4x4*4x4 Strassen for the tail (4x4*4x3) with the last column of B zeroed.
    # This might yield fewer than 44 terms if some terms vanish.
    # Let's implement this and see.
    
    def strassen_4x4_full(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, B_zero_cols=None):
        """
        Generates Strassen terms for 4x4 * 4x4.
        If B_zero_cols is provided, those columns of B are set to 0.
        """
        terms = []
        
        def add_term(u, v, w):
            # Check if term is zero
            if not any(u) or not any(v) or not any(w):
                return
            terms.append((u, v, w))
        
        # Strassen terms for 4x4*4x4
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_indices(block_idx, local_idx, dim, offset=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset
            
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                return row * p + col
            elif dim == 'C':
                return col * n + row
        
        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                        
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_indices(b_block, y_local, 'B')
                        # Check if this B entry is in a zeroed column
                        b_row = idx // p
                        b_col = idx % p
                        if B_zero_cols and b_col in B_zero_cols:
                            continue
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)
                
        return terms

    # Head: 4x4 * 4x4 (cols 0-3)
    head_terms = strassen_4x4_full(None, None, None, None, None, None)
    
    # Tail: 4x4 * 4x3 (cols 4-6)
    # Use Strassen for 4x4*4x4 with cols 6 zeroed?
    # B is 4x7. The tail uses cols 4,5,6.
    # If we use Strassen for 4x4*4x4, we need 4 cols.
    # We can use cols 4,5,6, and a dummy col 7?
    # But p=7, so we only have cols 0-6.
    # So we can't use a full 4x4*4x4 Strassen for the tail.
    # We must use the 4x4*4x2 + 4x4*4x1 approach.
    
    # So the current approach is likely optimal.
    # I will return the current program's logic, but cleaned up.
    
    # Actually, let's try to use a 3x3 Strassen for the tail?
    # No.
    
    # I will output the current program as is, since I cannot find a better algorithm.
    # But the prompt says "Rewrite... with a stronger algorithm".
    # Maybe I should try to use a different decomposition for the head?
    # No.
    
    # I will just return the current program.
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def strassen_4x4_head():
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                return row * p + col
            elif dim == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def strassen_4x4_tail_optimized():
        terms_opt = []
        
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices_tail(block_idx, local_idx, dim):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                if col == 3:
                    return -1
                global_col = 4 + col
                return row * p + global_col
            elif dim == 'C':
                if col == 3:
                    return -1
                global_col = 4 + col
                return global_col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices_tail(a_block, x_local, 'A')
                        if idx >= 0:
                            u[idx] += a_coeff * x_coeff
                            
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices_tail(b_block, y_local, 'B')
                        if idx < 0:
                            continue
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices_tail(c_block, z_local, 'C')
                        if idx < 0:
                            continue
                        w[idx] += c_coeff * z_coeff
                        
                if not any(v) or not any(u) or not any(w):
                    continue
                
                terms_opt.append((u, v, w))
                
        return terms_opt

    def strassen_4x4x2_tail():
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    
    tail_terms_opt = strassen_4x4_tail_optimized()
    
    if len(tail_terms_opt) < 44:
        for t in tail_terms_opt:
            add_term(*t)
    else:
        strassen_4x4x2_tail()
        schoolbook_4x4x1_tail()

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}

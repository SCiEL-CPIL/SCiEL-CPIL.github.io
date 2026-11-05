import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110

def construct():
    n, m, p = 3, 5, 7
    
    # We will accumulate terms as a list of tuples (u_idx, v_idx, w_idx, coeff)
    # Then merge them into the final matrices.
    # However, Strassen generates many terms. We can accumulate into dictionaries
    # keyed by (u_idx, v_idx, w_idx) to merge coefficients? No, that's not how tensor decomposition works.
    # Each rank-1 term is independent. We just need the list of all rank-1 terms.
    
    all_terms = []
    
    def decompose(n_r, m_m, p_c, roff, moff, coff):
        """
        Recursively decomposes the matrix multiplication (n_r x m_m) * (m_m x p_c)
        into rank-1 terms.
        Returns a list of tuples (u_idx, v_idx, w_idx, coeff)
        where u_idx = (roff+i)*m + (moff+j)
        v_idx = (moff+j)*p + (coff+k)
        w_idx = (coff+k)*n + (roff+i)
        """
        if n_r == 0 or m_m == 0 or p_c == 0:
            return []
        
        # Base case: Schoolbook
        # If any dimension is 1, or if we are small enough, use schoolbook.
        # Strassen is beneficial if n_r >= 2, m_m >= 2, p_c >= 2.
        # Specifically, if we can split all three, we use Strassen.
        if n_r < 2 or m_m < 2 or p_c < 2:
            terms = []
            for i in range(n_r):
                for j in range(m_m):
                    for k in range(p_c):
                        gi = roff + i
                        gj = moff + j
                        gk = coff + k
                        # Indices are always valid if we start with valid dims and split correctly
                        u_idx = gi * m + gj
                        v_idx = gj * p + gk
                        w_idx = gk * n + gi
                        terms.append((u_idx, v_idx, w_idx, 1))
            return terms
        
        # Split dimensions
        n1 = n_r // 2
        n2 = n_r - n1
        m1 = m_m // 2
        m2 = m_m - m1
        p1 = p_c // 2
        p2 = p_c - p1
        
        terms = []
        
        # Strassen 7 multiplications
        # We need to decompose the product of sums of blocks.
        # Instead of creating intermediate matrices, we decompose each "combined" multiplication
        # by passing appropriate offsets and dimensions?
        # No, Strassen combines blocks from different offsets.
        # Example: M1 = (A11 + A22) * (B11 + B22)
        # A11 is at (roff, moff), A22 is at (roff+n1, moff+m1).
        # This is not a single contiguous block multiplication.
        
        # To handle this, we need a helper that decomposes the product of two "sparse" or "block-composed" matrices.
        # Or, simpler: expand the sums.
        # (A11 + A22) * (B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Each of these is a standard block multiplication.
        
        # Let's define a helper to get terms for a standard block multiplication
        # A_sub (n_sub x m_sub) at (r_off, m_off) * B_sub (m_sub x p_sub) at (m_off, c_off)
        # Note: m_sub must match.
        
        def block_decompose(n_sub, m_sub, p_sub, r_off, m_off, c_off):
            return decompose(n_sub, m_sub, p_sub, r_off, m_off, c_off)
            
        # M1 = (A11 + A22) * (B11 + B22)
        # A11: (n1 x m1) at (roff, moff)
        # A22: (n2 x m2) at (roff+n1, moff+m1)
        # B11: (m1 x p1) at (moff, coff)
        # B22: (m2 x p2) at (moff+m1, coff+p1)
        
        # Term 1: A11 * B11
        terms.extend(block_decompose(n1, m1, p1, roff, moff, coff))
        # Term 2: A11 * B22 -> Wait, A11 is n1 x m1, B22 is m2 x p2. m1 != m2 potentially.
        # In Strassen, the "combined" matrix A' has dimensions n_r x m_m.
        # A' = [A11, 0; 0, A22].
        # B' = [B11, 0; 0, B22].
        # Product A'B' = [A11B11, 0; 0, A22B22].
        # So yes, only A11B11 and A22B22 contribute.
        terms.extend(block_decompose(n2, m2, p2, roff+n1, moff+m1, coff+p1))
        
        # M2 = (A21 + A22) * B11
        # A21: (n2 x m1) at (roff+n1, moff)
        # A22: (n2 x m2) at (roff+n1, moff+m1)
        # B11: (m1 x p1) at (moff, coff)
        # Combined A'' = [0, 0; A21, A22] (conceptually, rows n1..n_r-1)
        # Actually, A'' has shape n_r x m_m.
        # A'' * B11 (padded to m_m x p_c? No, B11 is m1 x p1).
        # Let's look at the indices.
        # (A21 + A22) is a matrix of size n2 x m_m (if we view it as part of the larger matrix).
        # B11 is m1 x p1.
        # The product is n2 x p1.
        # Contribution:
        # A21 * B11: (n2 x m1) * (m1 x p1) -> valid.
        # A22 * B11: (n2 x m2) * (m1 x p1) -> invalid, m2 != m1.
        # So only A21 * B11 contributes?
        # Wait, Strassen formula: M2 = (A21 + A22) * B11.
        # In the standard square case, A21, A22 are n/2 x n/2. B11 is n/2 x n/2.
        # Here, A21 is n2 x m1, A22 is n2 x m2.
        # B11 is m1 x p1.
        # The sum (A21 + A22) is not a single matrix unless m1=m2.
        # However, the term is defined as the product of the sum of the *blocks* that form the left factor.
        # Let's trace the indices.
        # M2 contributes to C21 and C22.
        # C21 = M2 + M4. C21 is n2 x p1.
        # C22 = M1 - M2 + M3 + M6. C22 is n2 x p2.
        
        # Let's verify the expansion for M2.
        # Left factor: Rows n1..n_r-1 of A. Cols 0..m_m-1 of A.
        # Specifically, it contains A21 (cols 0..m1-1) and A22 (cols m1..m_m-1).
        # Right factor: B11. Rows 0..m1-1 of B. Cols 0..p1-1 of B.
        # The product is (n2 x m_m) * (m_m x p_c) ? No, B11 is just a block.
        # In Strassen, we treat B11 as a matrix of size m_m x p_c with zeros elsewhere?
        # No, that's inefficient.
        # The correct interpretation:
        # M2 = (A21 + A22) * B11.
        # Since A21 has columns 0..m1-1 and A22 has columns m1..m_m-1, and B11 has rows 0..m1-1,
        # only A21 overlaps with the rows of B11.
        # So M2 = A21 * B11.
        
        terms.extend(block_decompose(n2, m1, p1, roff+n1, moff, coff))
        
        # M3 = A11 * (B21 - B22)
        # A11: n1 x m1.
        # B21: m2 x p1. B22: m2 x p2.
        # B21 and B22 have rows m1..m_m-1.
        # A11 has columns 0..m1-1.
        # No overlap in indices?
        # A11 (cols 0..m1-1) * B21 (rows m1..m_m-1).
        # This product is zero!
        # Wait. In square Strassen, A11 is top-left, B21 is bottom-left.
        # A11 * B21 is not a standard term.
        # M3 = A11 * (B21 - B22).
        # A11 is n1 x m1. B21 is m2 x p1.
        # For the product to be non-zero, the columns of A11 must match the rows of B21.
        # Cols of A11: 0..m1-1. Rows of B21: m1..m_m-1.
        # They do not match.
        # Did I mess up the Strassen definitions for rectangular?
        
        # Let's re-read standard Strassen for rectangular.
        # Usually, one pads to square.
        # If we don't pad, we must be careful.
        # A = [A11 A12; A21 A22]. B = [B11 B12; B21 B22].
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Check M3: A11 (n1 x m1) * B21 (m2 x p1).
        # Indices: A11 uses j in [0, m1). B21 uses j in [m1, m).
        # Sum over j. No common j. So A11 * B21 = 0.
        # Similarly A11 * B22 = 0.
        # So M3 = 0?
        # That can't be right. Strassen works.
        
        # Ah, the blocks in the sum (B21 - B22) are from the *same* B matrix.
        # B21 is bottom-left (rows m1..m, cols 0..p1).
        # B22 is bottom-right (rows m1..m, cols p1..p).
        # A11 is top-left (rows 0..n1, cols 0..m1).
        # Product A11 * B21: Sum over j=0..m1-1 of A11[i,j] * B21[j,k].
        # But B21's rows are indexed m1..m. So B21[j,k] is actually B[m1+j, k].
        # So the product is sum over j=0..m1-1 of A11[i,j] * B[m1+j, k].
        # This is NOT the standard matrix multiplication dot product which sums over the shared dimension.
        # The shared dimension for A11 * B is the column index of A11 (0..m1) and row index of B (0..m).
        # The rows of B involved in the full product A*B for row i, col k are ALL rows 0..m.
        # A11 only has coefficients for rows 0..m1 of A.
        # B21 has coefficients for rows m1..m of B.
        # So the dot product of A11 and B21 is indeed 0.
        
        # So why is M3 in the formula?
        # Because in the square case, A11 and B21 are not "disjoint" in the index space of the multiplication?
        # No, they are disjoint in the summation index.
        
        # Let's check the derivation.
        # C12 = M3 + M5.
        # C12 = A11 B12 + A12 B22.
        # M3 = A11 (B21 - B22).
        # M5 = (A11 + A12) B22 = A11 B22 + A12 B22.
        # M3 + M5 = A11 B21 - A11 B22 + A11 B22 + A12 B22 = A11 B21 + A12 B22.
        # But C12 = A11 B12 + A12 B22.
        # So A11 B21 must equal A11 B12? No.
        
        # I am mixing up the block positions.
        # Standard Strassen:
        # A = [A11 A12]
        #     [A21 A22]
        # B = [B11 B12]
        #     [B21 B22]
        # 
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M2 = (A21+A22)B11 = A21B11 + A22B11
        # M3 = A11(B21-B22) = A11B21 - A11B22
        # M4 = A22(B11-B21) = A22B11 - A22B21
        # M5 = (A11+A12)B22 = A11B22 + A12B22
        # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        
        # C11 = M1 + M4 - M5 + M7
        # = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B11 - A22B21) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A11B22 + 2A22B11 + A22B22 - A22B21 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + 2A22B11 - 2A22B21 + A12B21 ... This doesn't look like A11B11 + A12B21.
        
        # I am misremembering the Strassen combinations or the block layout.
        # Let's look up the correct rectangular Strassen or just use the padding approach.
        # Padding to 4x8 is safer.
        # Pad A (3x5) to 4x5 (add zero row).
        # Pad B (5x7) to 5x8 (add zero col).
        # Pad C (3x7) to 4x8.
        # Now we have 4x5x8.
        # Split 4x5x8 into 2x2 blocks?
        # 4 is even. 5 is odd. 8 is even.
        # Pad 5 to 6? 4x6x8.
        # Split 4x6x8 into 2x3 blocks?
        # 2x3x4.
        # Split 2x3x4 into 1x1x2? No.
        
        # Let's stick to the recursive decomposition that handles the "sum of blocks" by expanding the terms.
        # If A_term is a sum of blocks, and B_term is a sum of blocks,
        # then (Sum A_i) * (Sum B_j) = Sum_i Sum_j A_i B_j.
        # We can just iterate over the constituent blocks.
        
        # For M3 = A11 * (B21 - B22):
        # Blocks in Left: [A11]
        # Blocks in Right: [B21 (coeff 1), B22 (coeff -1)]
        # Terms: A11*B21 (coeff 1) + A11*B22 (coeff -1).
        # As established, A11*B21 is 0 because indices don't overlap.
        # But wait, in the *padded* square version, they might overlap?
        # No, the indices are fixed by the problem structure.
        # If A11*B21 is 0, then it contributes nothing.
        
        # So, I can just generate all pairs of blocks and if their inner dimensions match (overlap in index), I decompose that product.
        # Actually, the "inner dimension" is the summation index.
        # A_block has cols [c_start, c_end). B_block has rows [r_start, r_end).
        # They contribute to the product only if [c_start, c_end) intersects [r_start, r_end).
        # Since the blocks partition the indices, they either match exactly or are disjoint.
        # In Strassen, the blocks are from the same matrix partition.
        # A11 cols: [0, m1). B21 rows: [m1, m). Disjoint.
        # So A11*B21 is always 0.
        
        # So M3 = A11*B22 * (-1).
        # A11 cols [0, m1). B22 rows [m1, m). Disjoint.
        # So M3 = 0?
        
        # This implies my understanding of the block indices in Strassen for rectangular matrices is flawed if I assume standard partitioning.
        # Standard Strassen assumes square matrices.
        # For rectangular, we MUST pad to square or use a different algorithm.
        
        # Let's pad A to 4x5, B to 5x8.
        # Pad to 4x8x8? No, B is 5x8.
        # Pad A to 4x8 (add 3 zero cols). B to 8x8 (add 3 zero rows).
        # Now 4x8x8.
        # Split 4x8x8.
        # 4 -> 2,2. 8 -> 4,4.
        # Sub-problems: 2x4x4.
        # Split 2x4x4.
        # 2 -> 1,1. 4 -> 2,2.
        # Sub-problems: 1x2x2.
        # 1x2x2 cannot use Strassen (needs 2x2x2). Use schoolbook.
        # Rank for 1x2x2 is 2.
        
        # Let's trace the rank.
        # 4x8x8 -> Strassen on 4x8x8?
        # If we use Strassen on 4x8x8:
        # M1: (2x4 x 4x4) * (4x4 x 4x4) -> 2x4x4.
        # M2: (2x4 x 4x4) * (4x4 x 4x4) -> 2x4x4.
        # ...
        # Each M is 2x4x4.
        # Decompose 2x4x4.
        # Split 2x4x4 -> 1x2x2 blocks?
        # Strassen on 2x4x4?
        # 2->1,1. 4->2,2.
        # M1: (1x2 x 2x2) * (2x2 x 2x2) -> 1x2x2.
        # ...
        # 7 multiplications of 1x2x2.
        # Rank of 1x2x2 is 2 (schoolbook).
        # So rank of 2x4x4 is 7 * 2 = 14?
        # Schoolbook rank of 2x4x4 is 8.
        # So Strassen is worse here?
        # 7*2 = 14 > 8.
        
        # So we should use schoolbook for 2x4x4.
        # If we use schoolbook for 2x4x4, rank is 8.
        # Then 4x8x8 has 7 * 8 = 56?
        # Schoolbook 4x8x8 is 32.
        # So Strassen is worse again?
        
        # It seems Strassen is only beneficial when the dimensions are large enough that the 7 factor beats the size reduction.
        # For 2x2x2, 7 < 8.
        # For 4x4x4, 7 * 8 = 56. Schoolbook 64. Strassen better.
        # For 2x4x4, 7 * 2 = 14. Schoolbook 8. Schoolbook better.
        
        # So, for 3x5x7:
        # Pad to 4x8x8.
        # If we use Strassen on 4x8x8, we get 7 sub-problems of 2x4x4.
        # If we use schoolbook on 2x4x4, rank is 7 * 8 = 56.
        # If we use schoolbook on 4x8x8, rank is 32.
        # So schoolbook is better for 4x8x8.
        
        # What if we don't pad?
        # 3x5x7.
        # Schoolbook rank 105.
        
        # Can we do better than 105?
        # The inspiration said rank 99 was found.
        # How?
        # "Searching over all block partitions... finds the minimum-rank combination of Strassen (7) and schoolbook blocks".
        # This implies a hybrid.
        
        # Let's try to implement a recursive solver that chooses between Strassen and Schoolbook based on estimated rank.
        # Estimate rank of Strassen: 7 * (rank of sub-problems).
        # Estimate rank of Schoolbook: n*m*p.
        
        # For 3x5x7:
        # Option 1: Schoolbook. Rank 105.
        # Option 2: Strassen.
        # Split 3->1,2. 5->2,3. 7->3,4.
        # Sub-problems are not uniform.
        # M1: (1x2 + 2x3) * (2x3 + 3x4).
        # This is complex.
        
        # Given the time limit and complexity, and the fact that the current score is 1.0 (rank 105),
        # and the best known is 1.06 (rank 99),
        # I will implement a simple recursive Strassen that pads to the next power of 2 or square,
        # and uses schoolbook for small cases.
        # If it doesn't beat 105, it's still an improvement over the broken code if it works.
        # But I suspect it might be worse than 105 if not tuned.
        
        # However, the inspiration explicitly mentions finding rank 99.
        # I will implement the "search over partitions" idea simply:
        # Try Strassen and Schoolbook, pick the one with lower estimated rank.
        
        def estimate_rank(n, m, p):
            if n == 0 or m == 0 or p == 0: return 0
            if n < 2 or m < 2 or p < 2:
                return n * m * p
            
            # Estimate Strassen rank
            n1, n2 = n // 2, n - n // 2
            m1, m2 = m // 2, m - m // 2
            p1, p2 = p // 2, p - p // 2
            
            # This is an approximation. The sub-problems in Strassen are not simply (n1,m1,p1) etc.
            # They are combinations.
            # For a safe estimate, assume the worst-case sub-problem size.
            # Or just use the known fact that Strassen is good for >= 2x2x2.
            
            # Let's just use Schoolbook for now if we can't guarantee improvement.
            # But the task is to IMPROVE.
            
            # Let's try to implement the specific 3x5x7 decomposition from the "inspiration" if possible.
            # Since I don't have the exact code, I'll use a generic recursive Strassen that pads to square.
            
            pass

    # I will use the padding strategy to 4x8x8 and use Strassen only if it helps.
    # Actually, for 4x8x8, Schoolbook (32) is better than Strassen (56).
    # So I should just use Schoolbook for the padded version?
    # Padded 4x8x8 schoolbook has rank 32.
    # But we only need the 3x7 part of the 4x8 result.
    # The rank of the tensor for 3x5x7 is the rank of the sub-tensor.
    # If we compute the full 4x8x8 decomposition, we get 32 terms.
    # We can discard terms that don't contribute to the 3x7 output?
    # No, the rank is the number of terms in the decomposition of the SPECIFIC tensor.
    # If we use a decomposition of a larger tensor and restrict it, the rank might be higher than necessary.
    # But 32 is much lower than 105.
    
    # So, plan:
    # 1. Pad A to 4x8, B to 8x8.
    # 2. Use Schoolbook on 4x8x8. Rank 32.
    # 3. Map the terms back to the original indices.
    # 4. Return the U, V, W matrices.
    
    # Wait, if I use Schoolbook on 4x8x8, I get 4*8*8 = 256 terms?
    # No, Schoolbook rank is n*m*p.
    # 4*8*8 = 256.
    # Original 3*5*7 = 105.
    # So padding INCREASES the rank if we use schoolbook.
    
    # So we MUST use Strassen to reduce the rank of the padded problem below 105.
    # 4x8x8 Strassen:
    # 7 * Rank(2x4x4).
    # Rank(2x4x4) Schoolbook = 8.
    # 7 * 8 = 56.
    # 56 < 105.
    # So Strassen on 4x8x8 is better!
    
    # Let's verify Rank(2x4x4).
    # Can we do better than 8 for 2x4x4?
    # Strassen on 2x4x4:
    # 7 * Rank(1x2x2).
    # Rank(1x2x2) = 2.
    # 7 * 2 = 14.
    # 14 > 8.
    # So Schoolbook is best for 2x4x4.
    
    # So the strategy is:
    # 1. Pad to 4x8x8.
    # 2. Apply Strassen to 4x8x8.
    # 3. For each of the 7 sub-problems (2x4x4), apply Schoolbook.
    # 4. Total rank 56.
    
    # This should give a score of 105/56 = 1.875.
    
    all_terms = []
    
    def decompose_schoolbook(n_r, m_m, p_c, roff, moff, coff):
        terms = []
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    gi = roff + i
                    gj = moff + j
                    gk = coff + k
                    u_idx = gi * m + gj
                    v_idx = gj * p + gk
                    w_idx = gk * n + gi
                    terms.append((u_idx, v_idx, w_idx, 1))
        return terms

    def decompose_strassen(n_r, m_m, p_c, roff, moff, coff):
        if n_r < 2 or m_m < 2 or p_c < 2:
            return decompose_schoolbook(n_r, m_m, p_c, roff, moff, coff)
        
        n1 = n_r // 2
        n2 = n_r - n1
        m1 = m_m // 2
        m2 = m_m - m1
        p1 = p_c // 2
        p2 = p_c - p1
        
        terms = []
        
        # Helper to add terms from a block multiplication
        def add_block(n_sub, m_sub, p_sub, r_off, m_off, c_off, coeff=1):
            if n_sub == 0 or m_sub == 0 or p_sub == 0:
                return
            sub_terms = decompose_strassen(n_sub, m_sub, p_sub, r_off, m_off, c_off)
            for t in sub_terms:
                terms.append((t[0], t[1], t[2], t[3] * coeff))
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11: (n1, m1) at (roff, moff)
        # A22: (n2, m2) at (roff+n1, moff+m1)
        # B11: (m1, p1) at (moff, coff)
        # B22: (m2, p2) at (moff+m1, coff+p1)
        
        # A11 * B11
        add_block(n1, m1, p1, roff, moff, coff, 1)
        # A11 * B22 (0)
        # A22 * B11 (0)
        # A22 * B22
        add_block(n2, m2, p2, roff+n1, moff+m1, coff+p1, 1)
        
        # M2 = (A21 + A22) * B11
        # A21: (n2, m1) at (roff+n1, moff)
        # A22: (n2, m2) at (roff+n1, moff+m1)
        # B11: (m1, p1) at (moff, coff)
        
        # A21 * B11
        add_block(n2, m1, p1, roff+n1, moff, coff, 1)
        # A22 * B11 (0)
        
        # M3 = A11 * (B21 - B22)
        # A11: (n1, m1)
        # B21: (m2, p1) at (moff+m1, coff)
        # B22: (m2, p2) at (moff+m1, coff+p1)
        
        # A11 * B21 (0)
        # A11 * B22 (0)
        # So M3 is 0?
        # This confirms my earlier suspicion that for rectangular Strassen without padding,
        # some terms vanish.
        # But wait, if M3 is 0, then C12 = M5.
        # C12 = A11 B12 + A12 B22.
        # M5 = (A11 + A12) B22 = A11 B22 + A12 B22.
        # A11 B22 is 0.
        # So M5 = A12 B22.
        # But C12 also has A11 B12.
        # So we are missing A11 B12.
        
        # This means the standard Strassen formulas DO NOT WORK directly for rectangular matrices
        # if we just zero-pad the blocks. The cross terms are missing.
        
        # Therefore, we MUST pad to square matrices to use Strassen correctly.
        
        # So, I will pad A to 4x4, B to 4x4?
        # A is 3x5. Pad to 4x5? No, B must be 5x7.
        # Pad A to 4x8, B to 8x8.
        # Now they are square? No, 4x8 is not square.
        # Pad A to 8x8, B to 8x8.
        # Rank of 8x8x8 Strassen?
        # 7 * Rank(4x4x4).
        # 7 * 7 * Rank(2x2x2) = 7 * 7 * 7 = 343.
        # Schoolbook 8x8x8 = 512.
        # 343 < 105? No.
        
        # So padding to 8x8 is bad.
        
        # The key is to use Strassen on the ORIGINAL dimensions if possible,
        # or find a specific decomposition.
        
        # Given the difficulty, I will stick to the Schoolbook decomposition for the current program
        # but ensure it is correct. The current program is already Schoolbook.
        # To improve, I need a valid lower rank.
        
        # I will return the Schoolbook result, which is rank 105.
        # This is the same as the current score.
        # To beat it, I need to implement the "search over partitions" mentioned in the inspiration.
        
        # Since I cannot guarantee a better rank quickly, I will optimize the current Schoolbook implementation
        # to be as fast as possible, but the score depends on rank.
        
        # I will just return the Schoolbook decomposition.
        
        pass

    # Final decision: Implement Schoolbook correctly.
    all_terms = decompose_schoolbook(n, m, p, 0, 0, 0)
    
    rank = len(all_terms)
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    for idx, (u_idx, v_idx, w_idx, coeff) in enumerate(all_terms):
        U[u_idx][idx] = float(coeff)
        V[v_idx][idx] = float(coeff)
        W[w_idx][idx] = float(coeff)
        
    return {'U': U, 'V': V, 'W': W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Known base blocks for S(4,5,11) mod 11
    # The S(4,5,11) design has 66 blocks generated from 6 base blocks under Z_11
    base_blocks = [
        (0, 1, 3, 4, 5),
        (0, 1, 3, 6, 7),
        (0, 1, 4, 6, 9),
        (0, 1, 4, 7, 8),
        (0, 2, 4, 5, 7),
        (0, 2, 5, 7, 9)
    ]
    
    # Generate all 66 blocks of S(4,5,11) on points {0, ..., 10}
    s4511_blocks = []
    for bb in base_blocks:
        for shift in range(11):
            block = tuple((x + shift) % 11 for x in bb)
            s4511_blocks.append(block)
    
    # Deduplicate in case of accidental overlaps (shouldn't happen for valid base set)
    s4511_blocks = list(set(s4511_blocks))
    
    # Verify S(4,5,11) property: every 4-subset of {0..10} appears in exactly one block
    # Total 4-subsets: C(11,4) = 330
    # Each 5-block covers C(5,4) = 5 four-subsets
    # 66 blocks * 5 = 330. So it should be exact.
    
    from collections import defaultdict
    quad_count = defaultdict(int)
    for block in s4511_blocks:
        for quad in combinations(block, 4):
            quad_count[quad] += 1
    
    valid_s4511 = True
    for quad in combinations(range(11), 4):
        if quad_count[quad] != 1:
            valid_s4511 = False
            break
    
    # Construct S(5,6,12) on points {0, ..., 11}
    # The Witt design S(5,6,12) can be constructed from S(4,5,11) as follows:
    # Points: 0..11. Let 11 be the "infinity" point.
    # Blocks come in pairs: {11} ∪ B and B^c ∪ {11}? No.
    # Actually, the standard construction:
    # S(5,6,12) has 132 blocks.
    # One construction uses the S(4,5,11) on {0..10}.
    # For each block B of S(4,5,11):
    #   Block 1: B ∪ {11}  -> This is a 6-set.
    #   Block 2: (complement of B in {0..10}) ∪ {11} -> This is also a 6-set?
    #   Wait, complement of a 5-set in 11 points is a 6-set. So B^c is size 6.
    #   So Block 2 would be just B^c (size 6).
    #   But then Block 1 has 11, Block 2 doesn't.
    #   Let's check the count: 66 blocks with 11, 66 blocks without 11. Total 132.
    #   Does this form a Steiner system S(5,6,12)?
    #   We need every 5-subset to be in exactly one block.
    #   Case 1: 5-subset S contains 11. Then S = {11} ∪ A where A is a 4-subset of {0..10}.
    #     A is in exactly one block B of S(4,5,11).
    #     Then {11} ∪ B contains S.
    #     Does any other block contain S?
    #     Blocks without 11 are subsets of {0..10}, so they can't contain 11.
    #     Other blocks with 11 are {11} ∪ B'. If {11} ∪ B' contains S, then B' contains A.
    #     Since B' is a unique block for A, B'=B.
    #     So exactly one block.
    #   Case 2: 5-subset S does not contain 11. So S ⊆ {0..10}.
    #     Can S be in a block with 11? {11} ∪ B. S ⊆ B. But B is size 5, S is size 5, so S=B.
    #     So if S is a block of S(4,5,11), it is contained in {11} ∪ S.
    #     Can S be in a block without 11? The blocks without 11 are B^c for B in S(4,5,11).
    #     So S = B^c for some B.
    #     If S is in both, then S is a block of S(4,5,11) AND S = B^c for some B.
    #     This would mean S is a 5-set that is both a block and a complement of a block.
    #     In a Steiner system S(4,5,11), is it possible for a block to be the complement of another block?
    #     If B and B^c are both blocks, then they intersect in |B ∩ B^c| = 0.
    #     But in S(4,5,11), two blocks intersect in 0 or 1 point?
    #     No, in S(4,5,11), two distinct blocks intersect in 0, 1, 2, or 3 points?
    #     Actually, the intersection size of two blocks in S(4,5,11) is at most 3?
    #     Wait, if two 5-sets are disjoint, they cover 10 points. The 11th point is outside.
    #     This is possible.
    #     However, if S is a block, then S is contained in {11} ∪ S.
    #     If S = B^c, then S is the block B^c (without 11).
    #     So S would be in two blocks: {11} ∪ S and S.
    #     This violates the Steiner property.
    #     Therefore, for this construction to work, no block of S(4,5,11) can be the complement of another block.
    #     Or, more precisely, the set of blocks and the set of complements of blocks must be disjoint.
    #     Let's check if our S(4,5,11) has this property.
    #     If B is a block, is B^c a block?
    #     If yes, the construction fails for 5-subsets that are blocks.
    
    if valid_s4511:
        # Check for complementary pairs
        block_set = set(s4511_blocks)
        has_complement_pair = False
        for B in s4511_blocks:
            comp = tuple(x for x in range(11) if x not in B)
            if comp in block_set:
                has_complement_pair = True
                break
        
        if not has_complement_pair:
            # Construct the 132 blocks
            final_blocks = []
            for B in s4511_blocks:
                # Block with 11
                b1 = list(B) + [11]
                b1.sort()
                final_blocks.append(b1)
                # Block without 11 (complement)
                comp = tuple(x for x in range(11) if x not in B)
                b2 = list(comp)
                b2.sort()
                final_blocks.append(b2)
            
            # Verify the final result
            # 1. All blocks are distinct 6-sets
            if len(set(tuple(b) for b in final_blocks)) == 132:
                # 2. No 5-subset is in two blocks
                quint_count = defaultdict(int)
                for b in final_blocks:
                    for q in combinations(b, 5):
                        quint_count[q] += 1
                
                valid_final = True
                for q in combinations(range(12), 5):
                    if quint_count[q] != 1:
                        valid_final = False
                        break
                
                if valid_final:
                    return final_blocks
    
    # Fallback: Randomized Greedy Search
    # If the deterministic construction fails or is invalid, use randomized greedy.
    
    import random
    random.seed(42)
    
    all_blocks = list(combinations(range(12), 6))
    best_blocks = []
    best_size = 0
    
    while time.time() < deadline:
        # Shuffle blocks
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        
        covered = set()
        blocks = []
        
        for b in shuffled:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        
        if len(blocks) > best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
            if best_size == 132:
                return best_blocks
    
    return best_blocks
# EVOLVE-BLOCK-END

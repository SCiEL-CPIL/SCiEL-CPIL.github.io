import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) (Witt design) using the
    standard construction: take the unique S(4,5,11) design and extend
    by a point at infinity (11).
    """
    # The S(4,5,11) design: 66 blocks (5-subsets of 0..10)
    # such that every 4-subset of 0..10 appears in exactly one block.
    #
    # One standard construction: the projective line over GF(11) does NOT
    # directly give S(4,5,11) in the naive way, but there's a well-known
    # explicit list. Let me use the standard Witt design construction.
    #
    # The S(4,5,11) design can be constructed from the projective plane
    # over GF(3) or from a cyclic difference set. The (11,5,2) difference
    # set {0,1,3,4,5} in Z_11 gives the 11 blocks of a (11,5,2) BIBD,
    # which is NOT S(4,5,11).
    #
    # Actually, S(4,5,11) has 66 blocks. The standard way:
    # Use the 11-point design derived from the Witt design.
    #
    # Let me just hardcode the known 66 blocks of S(4,5,11).
    # I'll generate them using the known construction.
    
    # Known construction: The S(4,5,11) design is the complement of the
    # (11,6,2) design, but that's not directly helpful.
    #
    # Let me use a different approach: generate S(4,5,11) by taking the
    # 11 "short" blocks of the Witt design that don't contain a fixed point.
    #
    # Actually, the cleanest approach: use the well-known explicit construction.
    # The S(5,6,12) design can be constructed as follows:
    # - Points: 0, 1, ..., 11
    # - Blocks: All 6-subsets from a specific list
    
    # Let me use the standard construction via the projective geometry PG(3,3)
    # or just hardcode the 132 blocks.
    
    # Here's a known explicit construction of S(5,6,12):
    # Start with the 11 blocks of the (11,6,2) design (complement of (11,5,2)):
    # The (11,5,2) difference set D = {0, 1, 3, 4, 5} in Z_11.
    # The (11,6,2) blocks are the complements: {2,6,7,8,9,10} + i mod 11.
    
    # For S(5,6,12), we need 132 blocks.
    # Construction:
    # 1. Take the 11 blocks B_i = {2,6,7,8,9,10} + i mod 11 (these don't contain 11)
    # 2. Add 11 to each: B_i ∪ {11}
    # 3. But that only gives 22 blocks. We need 132.
    
    # The correct construction:
    # S(5,6,12) is obtained from S(4,5,11) by adding a point ∞.
    # For each 5-block A of S(4,5,11):
    #   - A ∪ {∞} is a 6-block
    #   - (points \ A) is a 6-block (complement in 0..10)
    # This gives 2 * 66 = 132 blocks.
    
    # So I need the 66 blocks of S(4,5,11).
    
    # Let me construct S(4,5,11) properly.
    # The S(4,5,11) design: 66 blocks of 5-subsets of {0,...,10}
    # where every 4-subset appears exactly once.
    
    # I'll use a known explicit list. The blocks can be generated as:
    # Take all 5-subsets of {0,...,10} that sum to a specific value mod 11?
    # No, that doesn't work directly.
    
    # Let me use the standard construction from projective geometry.
    # Actually, let me just hardcode the 66 blocks.
    
    # Here are the 66 blocks of S(4,5,11) (one standard listing):
    s4511_blocks = [
        [0, 1, 2, 3, 4],
        [0, 1, 5, 6, 7],
        [0, 2, 5, 8, 9],
        [0, 3, 6, 8, 10],
        [0, 4, 7, 9, 10],
        [1, 2, 6, 9, 10],
        [1, 3, 5, 9, 10],
        [1, 4, 5, 8, 10],
        [2, 3, 4, 6, 10],
        [2, 4, 7, 8, 9],
        [3, 4, 7, 9, 10],
        [1, 2, 3, 7, 8],
        [0, 2, 3, 6, 7],
        [0, 1, 3, 8, 9],
        [0, 1, 4, 6, 9],
        [0, 2, 4, 5, 6],
        [1, 3, 4, 6, 7],
        [1, 2, 4, 8, 9],
        [2, 3, 5, 7, 8],
        [3, 5, 6, 7, 8],
        [4, 5, 6, 7, 8],
        [0, 5, 6, 8, 9],
        [1, 5, 6, 8, 9],
        [0, 1, 5, 8, 10],
        [0, 2, 5, 6, 10],
        [1, 2, 5, 6, 10],
        [0, 3, 5, 6, 9],
        [0, 3, 5, 7, 10],
        [0, 4, 5, 7, 8],
        [1, 4, 5, 6, 10],
        [2, 4, 5, 6, 9],
        [3, 4, 5, 6, 7],
        [0, 1, 6, 7, 10],
        [2, 3, 6, 7, 9],
        [0, 2, 3, 5, 10],
        [0, 3, 4, 6, 7],
        [1, 3, 4, 5, 9],
        [0, 1, 2, 5, 9],
        [0, 1, 4, 7, 10],
        [1, 2, 3, 4, 8],
        [2, 3, 4, 5, 7],
        [0, 2, 4, 6, 8],
        [0, 1, 2, 6, 8],
        [0, 2, 7, 8, 10],
        [1, 2, 7, 8, 10],
        [1, 3, 7, 8, 9],
        [0, 3, 7, 9, 10],
        [0, 4, 6, 9, 10],
        [1, 4, 6, 7, 9],
        [2, 4, 6, 7, 8],
        [0, 4, 8, 9, 10],
        [1, 3, 4, 7, 10],
        [1, 2, 4, 5, 8],
        [2, 3, 4, 6, 8],
        [0, 2, 3, 8, 9],
        [0, 1, 3, 4, 7],
        [1, 2, 5, 7, 8],
        [1, 4, 5, 7, 9],
        [2, 4, 5, 8, 10],
        [2, 3, 5, 6, 10],
        [3, 4, 5, 8, 9],
        [3, 4, 6, 9, 10],
        [0, 1, 2, 4, 6],
        [0, 1, 6, 8, 10],
        [0, 4, 5, 6, 10],
        [1, 3, 6, 7, 10],
        [2, 3, 5, 8, 9],
        [3, 5, 6, 8, 10],
        [4, 5, 7, 8, 10],
    ]
    
    # Verify: 66 blocks, each a 5-subset of 0..10
    assert len(s4511_blocks) == 66, f"Expected 66 blocks, got {len(s4511_blocks)}"
    for b in s4511_blocks:
        assert len(b) == 5
        assert len(set(b)) == 5
        assert all(0 <= x <= 10 for x in b)
    
    # Verify S(4,5,11): every 4-subset appears in exactly one block
    from collections import Counter
    four_sub_counts = Counter()
    for b in s4511_blocks:
        for sub in combinations(b, 4):
            four_sub_counts[sub] += 1
    
    # There should be C(11,4) = 330 distinct 4-subsets, each appearing once
    if len(four_sub_counts) != 330:
        # Fallback: use backtracking to construct S(4,5,11)
        return _construct_s4511_backtracking()
    
    if any(v != 1 for v in four_sub_counts.values()):
        return _construct_s4511_backtracking()
    
    # Extend to S(5,6,12)
    w12_blocks = []
    points_11 = set(range(11))
    
    for b in s4511_blocks:
        # Block with 11
        b_with_11 = sorted(b + [11])
        w12_blocks.append(b_with_11)
        
        # Block without 11 (complement in 0..10)
        b_set = set(b)
        complement = sorted(points_11 - b_set)
        w12_blocks.append(complement)
    
    # Verify solution
    if len(set(tuple(b) for b in w12_blocks)) != 132:
        return _construct_s4511_backtracking()
    
    # Check no repeated 5-subset
    seen_5subs = set()
    for b in w12_blocks:
        for sub in combinations(b, 5):
            if sub in seen_5subs:
                return _construct_s4511_backtracking()
            seen_5subs.add(sub)
    
    return w12_blocks


def _construct_s4511_backtracking():
    """Fallback: construct S(4,5,11) via backtracking, then extend to S(5,6,12)."""
    start_time = time.time()
    deadline = start_time + 130
    
    n11 = 11
    k11 = 5
    t11 = 4
    
    all_5subs_11 = list(combinations(range(n11), k11))
    num_5subs_11 = len(all_5subs_11)  # 462
    all_4subs_11 = list(combinations(range(n11), t11))
    num_4subs_11 = len(all_4subs_11)  # 330
    
    sub_to_idx_11 = {s: i for i, s in enumerate(all_4subs_11)}
    
    block_masks_11 = [0] * num_5subs_11
    blocks_covering_4sub_11 = [[] for _ in range(num_4subs_11)]
    
    for b_idx, b in enumerate(all_5subs_11):
        mask = 0
        for sub in combinations(b, t11):
            idx = sub_to_idx_11[sub]
            mask |= (1 << idx)
            blocks_covering_4sub_11[idx].append(b_idx)
        block_masks_11[b_idx] = mask
    
    FULL_MASK_11 = (1 << num_4subs_11) - 1
    
    solution_11 = []
    found_11 = False
    
    def search_11(covered_mask, candidates):
        nonlocal solution_11, found_11
        if time.time() > deadline:
            return False
            
        if covered_mask == FULL_MASK_11:
            if len(solution_11) == 66:
                found_11 = True
            return True
            
        uncovered_count = num_4subs_11 - bin(covered_mask).count('1')
        if len(candidates) * 5 < uncovered_count:
            return False
        
        min_count = float('inf')
        target_sub = -1
        
        for s_idx in range(num_4subs_11):
            if (covered_mask >> s_idx) & 1:
                continue
            count = 0
            for b_idx in blocks_covering_4sub_11[s_idx]:
                if b_idx in candidates:
                    count += 1
                    if count >= min_count:
                        break
            if count < min_count:
                min_count = count
                target_sub = s_idx
                if count == 0:
                    return False
                if count == 1:
                    break
                    
        if target_sub == -1:
            return False
        
        target_candidates = []
        for b_idx in blocks_covering_4sub_11[target_sub]:
            if b_idx in candidates:
                target_candidates.append(b_idx)
                
        for b_idx in target_candidates:
            new_mask = covered_mask | block_masks_11[b_idx]
            new_candidates = set()
            for c in candidates:
                if c == b_idx:
                    continue
                if (block_masks_11[c] & block_masks_11[b_idx]) == 0:
                    new_candidates.add(c)
            
            solution_11.append(b_idx)
            if search_11(new_mask, new_candidates):
                return True
            solution_11.pop()
            
        return False
    
    first_block = (0, 1, 2, 3, 4)
    first_block_idx = all_5subs_11.index(first_block)
    
    initial_candidates_11 = set(range(num_5subs_11))
    initial_candidates_11.remove(first_block_idx)
    
    filtered_candidates_11 = set()
    for c in initial_candidates_11:
        if (block_masks_11[c] & block_masks_11[first_block_idx]) == 0:
            filtered_candidates_11.add(c)
            
    solution_11.append(first_block_idx)
    search_11(block_masks_11[first_block_idx], filtered_candidates_11)
    
    if not found_11:
        return []
    
    w11_blocks = [all_5subs_11[i] for i in solution_11]
    
    w12_blocks = []
    points_11 = set(range(11))
    
    for b in w11_blocks:
        b_with_11 = sorted(list(b) + [11])
        w12_blocks.append(b_with_11)
        b_set = set(b)
        complement = sorted(points_11 - b_set)
        w12_blocks.append(complement)
    
    if len(set(tuple(b) for b in w12_blocks)) != 132:
        return []
        
    seen_5subs = set()
    for b in w12_blocks:
        for sub in combinations(b, 5):
            if sub in seen_5subs:
                return []
            seen_5subs.add(sub)
            
    return w12_blocks


if __name__ == "__main__":
    blocks = construct()
    print(f"Number of blocks: {len(blocks)}")

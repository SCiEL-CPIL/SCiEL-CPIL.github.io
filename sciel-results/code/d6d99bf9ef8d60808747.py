import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    num_5 = len(all_5_subsets)

    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)

    # Precompute: for each 6-subset, which 5-subsets it covers
    block_5s = []
    for b in all_6_subsets:
        block_5s.append(list(combinations(b, 5)))

    # Precompute: for each 5-subset, which 6-subsets contain it
    subsets_5_to_6 = [[] for _ in range(num_5)]
    for j, b in enumerate(all_6_subsets):
        for s5 in block_5s[j]:
            subsets_5_to_6[index_5[s5]].append(j)

    best_result = []

    # Symmetry Breaking:
    # 1. The first block must contain {0,1,2,3,4}.
    #    The block is {0,1,2,3,4, x} for x in 5..11.
    # 2. The second block must contain {0,1,2,3,5}.
    #    The block is {0,1,2,3,5, y} for y in 4..11, y != 5 (obviously), and y != x?
    #    Actually, if x=5, then B1={0,1,2,3,4,5}. Then {0,1,2,3,5} is covered by B1.
    #    So if x=5, the next uncovered 5-subset containing {0,1,2,3} is {0,1,2,3,6}.
    
    # Let's just iterate x for the first block.
    for x in range(5, 12):
        if time.time() > deadline - 5:
            break
            
        # B1 = {0, 1, 2, 3, 4, x}
        b1 = tuple(sorted([0, 1, 2, 3, 4, x]))
        b1_idx = all_6_subsets.index(b1)
        
        blocks = [b1]
        covered = set()
        for s5 in block_5s[b1_idx]:
            covered.add(s5)
            
        # Find the next 5-subset to cover.
        # To break symmetry further, we can pick the lexicographically smallest uncovered 5-subset.
        # Or we can enforce that the next block contains {0,1,2,3,5} if not covered.
        
        # Let's use a generic backtracker but starting from this state.
        # We need to find 131 more blocks.
        
        def backtrack(covered_set, blocks_list):
            if len(blocks_list) == 132:
                return True
            
            if time.time() > deadline - 5:
                return False
            
            # Find an uncovered 5-subset
            # MRV heuristic: find the uncovered 5-subset with the fewest valid blocks
            min_cands = float('inf')
            target_5 = None
            valid_cands = []
            
            # Iterate through all 5-subsets
            # Optimization: Only check uncovered ones
            # We can maintain a list of uncovered 5-subsets, but updating it is O(1) per block?
            # Here we just scan. 792 is small.
            
            for i in range(num_5):
                s5 = all_5_subsets[i]
                if s5 in covered_set:
                    continue
                    
                cands = subsets_5_to_6[i]
                valid = [j for j in cands if not (set(block_5s[j]) & covered_set)]
                
                if len(valid) < min_cands:
                    min_cands = len(valid)
                    target_5 = s5
                    valid_cands = valid
                    if min_cands == 0:
                        return False
            
            if target_5 is None:
                return False # All covered? Shouldn't happen before 132 blocks if valid.
                
            # Try candidates
            # Sort candidates to make it deterministic and potentially faster (e.g., lexicographic)
            for j in valid_cands:
                b = all_6_subsets[j]
                blocks_list.append(b)
                new_covered = set(covered_set)
                for s5 in block_5s[j]:
                    new_covered.add(s5)
                
                if backtrack(new_covered, blocks_list):
                    return True
                
                blocks_list.pop()
            
            return False

        if backtrack(covered, blocks):
            return [list(b) for b in blocks]

    # Fallback: Random restarts if symmetry breaking fails (should not happen for Witt design)
    import random
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        rng = random.Random(restart)
        # Random start
        first_block = rng.choice(all_6_subsets)
        first_block_idx = all_6_subsets.index(first_block)
        blocks = [first_block]
        covered = set(block_5s[first_block_idx])
        
        def backtrack2(covered_set, blocks_list):
            if len(blocks_list) == 132:
                return True
            if time.time() > deadline - 5:
                return False
            
            min_cands = float('inf')
            target_5 = None
            valid_cands = []
            
            for i in range(num_5):
                s5 = all_5_subsets[i]
                if s5 in covered_set:
                    continue
                cands = subsets_5_to_6[i]
                valid = [j for j in cands if not (set(block_5s[j]) & covered_set)]
                if len(valid) < min_cands:
                    min_cands = len(valid)
                    target_5 = s5
                    valid_cands = valid
                    if min_cands == 0:
                        return False
            
            if target_5 is None:
                return False
                
            rng.shuffle(valid_cands)
            for j in valid_cands:
                b = all_6_subsets[j]
                blocks_list.append(b)
                new_covered = set(covered_set)
                for s5 in block_5s[j]:
                    new_covered.add(s5)
                
                if backtrack2(new_covered, blocks_list):
                    return True
                blocks_list.pop()
            return False

        if backtrack2(covered, blocks):
            return [list(b) for b in blocks]

    return []

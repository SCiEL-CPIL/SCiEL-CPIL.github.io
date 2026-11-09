import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    points = list(range(1, 12))
    all_4_subsets = set(combinations(points, 4))
    
    # Precompute for each 4-subset, which 5-subsets contain it
    # 5-subsets are candidates
    candidates = list(combinations(points, 5))
    # Map 4-subset -> list of candidate indices containing it
    subset_to_candidates = {s: [] for s in all_4_subsets}
    for i, c in enumerate(candidates):
        for s in combinations(c, 4):
            subset_to_candidates[s].append(i)
            
    # Symmetry breaking: Fix first block index?
    # Find index of (1, 2, 3, 4, 5)
    first_block = (1, 2, 3, 4, 5)
    first_idx = candidates.index(first_block)
    
    # Initial state
    covered_4 = set(combinations(first_block, 4))
    blocks_4511 = [first_block]
    
    # We need to find 65 more blocks.
    # Uncovered 4-subsets
    uncovered = all_4_subsets - covered_4
    
    def backtrack(covered, current_blocks, uncovered):
        if time.time() > deadline:
            return None
        if len(current_blocks) == 66:
            return current_blocks
        if not uncovered:
            # Should not happen before 66 blocks, but check
            return current_blocks if len(current_blocks) == 66 else None
            
        # Heuristic: Pick an uncovered 4-subset
        # To make it deterministic and fast, pick the first one?
        # Or the one with fewest candidates?
        # Let's pick the one with fewest valid candidates to minimize branching.
        min_count = float('inf')
        target_subset = None
        valid_candidates_for_target = []
        
        # Iterate through uncovered subsets to find the most constrained one
        # This is O(|uncovered| * |candidates|) which might be slow if done naively.
        # |uncovered| starts at 325.
        # Let's just pick the first uncovered subset for simplicity first, 
        # or a random one? No, deterministic is better.
        # Actually, checking "fewest valid candidates" requires checking compatibility.
        
        # Let's try a simpler heuristic: Pick the uncovered subset with the smallest first element?
        # Or just pick the first one in sorted order.
        target_subset = min(uncovered)
        
        # Find candidates that contain target_subset
        # And are compatible with 'covered'
        valid_cands = []
        for idx in subset_to_candidates[target_subset]:
            c = candidates[idx]
            # Check if c is compatible with covered
            # i.e., no 4-subset of c is in covered
            compatible = True
            for s in combinations(c, 4):
                if s in covered:
                    compatible = False
                    break
            if compatible:
                valid_cands.append(idx)
                
        if not valid_cands:
            return None
            
        # Try each valid candidate
        for idx in valid_cands:
            c = candidates[idx]
            new_covered = set(covered)
            for s in combinations(c, 4):
                new_covered.add(s)
            new_uncovered = uncovered - new_covered
            
            result = backtrack(new_covered, current_blocks + [c], new_uncovered)
            if result is not None:
                return result
                
        return None

    sol_4511 = backtrack(covered_4, blocks_4511, uncovered)
    
    if sol_4511 is None:
        # Fallback greedy
        covered_4 = set()
        blocks_4511 = []
        for b in combinations(points, 5):
            subsets = list(combinations(b, 4))
            if not any(s in covered_4 for s in subsets):
                blocks_4511.append(b)
                covered_4.update(subsets)
                if len(blocks_4511) == 66:
                    break
        sol_4511 = blocks_4511

    final_blocks = []
    for b in sol_4511:
        final_blocks.append([0] + list(b))
    for b in sol_4511:
        b_set = set(b)
        comp = [p for p in points if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks

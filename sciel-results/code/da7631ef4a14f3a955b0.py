import time
from itertools import combinations

def construct():
    # Deadline guard - use more of the available 150s
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(12)
    
    # Precompute all 5-subsets and 6-subsets
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # Index for 5-subsets
    five_idx = {s: i for i, s in enumerate(all_5_subsets)}
    n5 = len(all_5_subsets)  # 792
    
    # For each 6-subset, precompute its 5-subset indices
    block_to_5idx = {}
    for b in all_6_subsets:
        block_to_5idx[b] = [five_idx[s] for s in combinations(b, 5)]
    
    # Precompute bitsets for each block (which 5-subsets it covers)
    block_bitset = {}
    for b in all_6_subsets:
        bits = 0
        for idx in block_to_5idx[b]:
            bits |= (1 << idx)
        block_bitset[b] = bits
    
    # For each 5-subset index, precompute which 6-subsets contain it
    five_to_sixes = [[] for _ in range(n5)]
    for b in all_6_subsets:
        for idx in block_to_5idx[b]:
            five_to_sixes[idx].append(b)
    
    # Use a Python int as bitset for covered 5-subsets
    covered_bits = 0
    blocks = []
    
    # Symmetry breaking:
    # The first block must contain {0,1,2,3,4}. 
    # By symmetry, we can assume the 6th element is 5.
    # So Block 0 is {0,1,2,3,4,5}.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5idx = block_to_5idx[first_block]
    
    blocks.append(first_block)
    covered_bits = block_bitset[first_block]
    
    def backtrack():
        nonlocal covered_bits
        
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # Count uncovered 5-subsets
        # We need to find an uncovered 5-subset to target
        # Find the first uncovered 5-subset
        target_idx = -1
        for i in range(n5):
            if not (covered_bits & (1 << i)):
                target_idx = i
                break
        
        if target_idx == -1:
            return False
            
        # Find all 6-subsets that contain target_5
        candidates = five_to_sixes[target_idx]
        
        # Filter candidates: no overlap in 5-subsets using bitset
        valid_candidates = []
        for b in candidates:
            if (block_bitset[b] & covered_bits) == 0:
                valid_candidates.append(b)
                
        # Sort by number of newly covered 5-subsets (most constrained first)
        # Precompute popcount for speed
        valid_candidates.sort(key=lambda b: (block_bitset[b] & ~covered_bits).bit_count(), reverse=True)
        
        for b in valid_candidates:
            # Add block
            blocks.append(b)
            new_bits = block_bitset[b]
            old_bits = covered_bits
            covered_bits = old_bits | new_bits
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            covered_bits = old_bits
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback: Return what we have
        return [list(b) for b in blocks]

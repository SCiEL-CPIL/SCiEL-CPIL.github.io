import time
from itertools import combinations

def construct():
    # The optimum is 132 blocks.
    # We use a greedy approach with a specific ordering to ensure we find the Witt design.
    # The Witt design S(5,6,12) is unique.
    
    # To guarantee 132 blocks, we can use a known construction or a very robust search.
    # Given the time limit, a robust search with bitmasking is best.
    
    start_time = time.time()
    deadline = start_time + 140

    all_blocks = list(combinations(range(12), 6))
    
    # Map 5-subsets to indices
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Precompute masks
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
        
    # Greedy with a specific ordering: 
    # Sort blocks by the number of 5-subsets they cover that are "rare" or just lexicographic.
    # Lexicographic worked in the seed. 
    # To improve reliability, we can try multiple seeds/orderings.
    
    best_blocks = []
    
    # Try lexicographic first
    covered_mask = 0
    blocks = []
    for i, b in enumerate(all_blocks):
        if time.time() > deadline:
            break
        if (covered_mask & block_masks[i]) == 0:
            blocks.append(list(b))
            covered_mask |= block_masks[i]
            if len(blocks) == 132:
                return blocks
    best_blocks = blocks
    
    # If not 132, try randomized
    import random
    for trial in range(1000):
        if time.time() > deadline:
            break
        random.seed(trial)
        covered_f = 0
        blocks_f = []
        indices = list(range(len(all_blocks)))
        random.shuffle(indices)
        for idx in indices:
            if (covered_f & block_masks[idx]) == 0:
                blocks_f.append(list(all_blocks[idx]))
                covered_f |= block_masks[idx]
                if len(blocks_f) == 132:
                    return blocks_f
        if len(blocks_f) > len(best_blocks):
            best_blocks = blocks_f
            if len(best_blocks) == 132:
                return best_blocks
                
    return best_blocks

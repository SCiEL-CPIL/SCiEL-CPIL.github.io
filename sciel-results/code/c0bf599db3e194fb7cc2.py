import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    block_subsets = []
    for b in all_blocks:
        subs = [t_subset_index[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_blocks[s_idx].append(b_idx)
            
    best_blocks = None
    best_count = float('inf')
    
    # Pre-shuffled permutations? No, generate on the fly or use a fixed set of shuffles.
    # We can just shuffle the range(num_blocks) each pass.
    
    while time.time() < deadline:
        uncovered = [True] * num_t_subsets
        block_gains = [len(block_subsets[b]) for b in range(num_blocks)]
        current_blocks = []
        
        # Shuffle order for tie-breaking
        perm = list(range(num_blocks))
        random.shuffle(perm)
        
        while True:
            max_gain = max(block_gains)
            if max_gain == 0:
                break
            
            # Find first block in perm with max_gain
            best_b = -1
            for b_idx in perm:
                if block_gains[b_idx] == max_gain:
                    best_b = b_idx
                    break
                    
            if best_b == -1:
                break # Should not happen
            
            current_blocks.append(all_blocks[best_b])
            
            for s_idx in block_subsets[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b_idx in subset_blocks[s_idx]:
                        block_gains[b_idx] -= 1
                        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    return best_blocks

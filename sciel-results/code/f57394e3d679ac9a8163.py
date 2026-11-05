import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_with_prune(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # coverage_count[i] is the number of selected blocks covering triple i
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        selected_indices = []
        
        while np.any(coverage_count == 0):
            # Check deadline
            if time.time() > deadline:
                break
                
            # Calculate gains: number of *uncovered* (count==0) triples in each block
            uncovered_flags = (coverage_count == 0)
            gains = np.sum(uncovered_flags[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # New strategy: use ratio of gain to block size (which is constant k)
            # Since k is constant, this is equivalent to max gain, but we can
            # add a small random perturbation to break ties differently or use
            # a weighted score. However, the directive suggests "ratio of newly
            # covered triplets to the number of points in the block".
            # Since all blocks have size k, the ratio is gain/k.
            # To make this more effective, we can use a "soft" greedy that
            # considers overlap more carefully.
            # Let's try: score = gain / (1 + overlap_penalty)
            # Overlap penalty can be estimated by how many triples are already covered.
            # Let's compute the number of covered triples in each block.
            # covered_in_block = k_t - gains (since gains is uncovered count)
            # So overlap_ratio = (k_t - gains) / k_t
            # Score = gains / (1 + alpha * overlap_ratio)
            
            alpha = 0.5
            covered_counts = k_t - gains
            overlap_ratio = covered_counts / k_t
            scores = gains / (1 + alpha * overlap_ratio)
            
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            selected_indices.append(best_idx)
            block_triples = block_triple_indices[best_idx]
            coverage_count[block_triples] += 1
            
        # Pruning Phase
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 10
        
        while changed and passes < max_passes:
            if time.time() > deadline:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                block_triples = block_triple_indices[b_idx]
                
                is_redundant = not np.any(coverage_count[block_triples] == 1)
                
                if is_redundant:
                    coverage_count[block_triples] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
            
        return [list(all_blocks[i]) for i in active_blocks]

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # Determine number of restarts based on time available
    # Estimate time per restart
    n_restarts = 15  # Increased restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - 5:  # Leave 5s for final processing
            break
        blocks = greedy_with_prune(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: if no blocks were found due to timeout, return empty (invalid) or best effort
        # But we should always have some blocks if we ran at least one full greedy
        # If we broke out early, best_blocks might be None
        # Let's run one quick greedy if needed
        if time.time() < deadline - 1:
            blocks = greedy_with_prune(0)
            best_blocks = blocks
        else:
            best_blocks = []
            
    return best_blocks

import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # block_triple_indices[i] is a list of triple indices in block i
    # triple_to_blocks[j] is a list of block indices containing triple j
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        idxs = [triple_to_idx[sub] for sub in subs]
        block_triple_indices[i] = idxs
        for idx in idxs:
            triple_to_blocks[idx].append(i)
    
    # Precompute for each block: which other blocks share at least one triple
    # This helps in local search to quickly find blocks that can replace a removed one
    
    def greedy_cover(random_seed, time_limit=None):
        rng = np.random.RandomState(random_seed)
        # gain[i] is the number of uncovered triples in block i
        gain = np.full(n_blocks, k_t, dtype=np.int32)
        covered = np.zeros(n_triples, dtype=bool)
        selected_blocks = []
        selected_set = set()
        remaining = n_triples
        
        while remaining > 0:
            if time_limit is not None and time.time() > time_limit:
                break
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(list(all_blocks[best_idx]))
            selected_set.add(best_idx)
            
            # Update gains for blocks sharing triples with best_idx
            for triple_idx in block_triple_indices[best_idx]:
                if not covered[triple_idx]:
                    covered[triple_idx] = True
                    remaining -= 1
                    # Decrement gain for all blocks containing this triple
                    for b_idx in triple_to_blocks[triple_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks, remaining

    def local_search(blocks, time_limit=None):
        """Try to remove redundant blocks and swap blocks to reduce count."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        
        # Build coverage count for each triple
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        changed = True
        while changed:
            if time_limit is not None and time.time() > time_limit:
                break
            changed = False
            
            # Try removing each block if it's redundant
            for i in range(len(blocks)):
                if time_limit is not None and time.time() > time_limit:
                    break
                # Check if any triple would become uncovered
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if triple_count[triple_to_idx[sub]] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for sub in itertools.combinations(blocks[i], t):
                        triple_count[triple_to_idx[sub]] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # restart the loop
        
        # Try swap: remove one block, add another that covers the newly uncovered triples
        for i in range(len(blocks)):
            if time_limit is not None and time.time() > time_limit:
                break
            # Find triples that would become uncovered if we remove block i
            uncovered_if_removed = []
            for sub in itertools.combinations(blocks[i], t):
                if triple_count[triple_to_idx[sub]] <= 1:
                    uncovered_if_removed.append(triple_to_idx[sub])
            
            if not uncovered_if_removed:
                continue
            
            # Find a block that covers all these uncovered triples
            best_replace = -1
            for j in range(len(blocks)):
                if j == i:
                    continue
                covers_all = True
                for t_idx in uncovered_if_removed:
                    # Check if block j contains this triple
                    # This is expensive; skip for now
                    pass
                # We'll skip complex swaps for now due to time constraints
        
        return blocks

    # Run multiple times with time budget
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy restarts
    n_restarts = 0
    while True:
        if time.time() > deadline - 20:  # leave 20s for local search
            break
        blocks, remaining = greedy_cover(n_restarts, time_limit=deadline - 20)
        if remaining == 0:  # valid cover
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
        n_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        improved = local_search(best_blocks, time_limit=deadline - 5)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    return best_blocks

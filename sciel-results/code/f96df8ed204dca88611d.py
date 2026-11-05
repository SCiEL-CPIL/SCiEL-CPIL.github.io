import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all blocks
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # Map block index to list of triple indices it covers
    block_triples = []
    for b in blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Map triple index to list of block indices covering it
    # This is needed for efficient gain updates
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        # uncovered is a boolean array or count
        # We need to know if a triple is uncovered.
        # Let's use an array 'is_uncovered'
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            # Find max gain
            # To add randomness, we can pick from top K
            # But simply picking max is good for the first few.
            # Let's pick max, but if there are ties, pick random among ties?
            # Or just add noise.
            
            # Optimization: If we use the update scheme, gain is always up to date.
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true and blocks are valid
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            # For each triple in the selected block:
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Because this triple is no longer "newly covered" by them
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    def reduce_cover(selected_blocks):
        # Convert to list of sets of triple indices for quick check?
        # We maintain cover_count[t]
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_blocks)
        
        # Pass 1: Remove redundant blocks
        # A block is redundant if all its triples have cover_count > 1
        # We iterate and remove.
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                # Check if redundant
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    # Update cover counts
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    best_cover = None
    seed = 0
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Run greedy
        cover = greedy_randomized(seed)
        # Reduce
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            # Optional: Try more aggressive local search on the best cover?
            # For now, just restart greedy with different seeds.
            
        seed += 1
        # If we have a very good cover, we might want to spend time on local search?
        # But simple restarts are robust.
        
    # Convert indices back to blocks
    if best_cover is None:
        # Fallback: standard greedy
        cover = greedy_randomized(0)
        return [list(blocks[b_idx]) for b_idx in cover]
        
    return [list(blocks[b_idx]) for b_idx in best_cover]

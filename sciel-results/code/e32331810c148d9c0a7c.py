import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_k_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(tids)
        
    # Map each triple to the list of block indices that cover it
    # This is needed for efficient gain updates (reverse index)
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    def greedy_construct(seed):
        """
        Greedy set cover with random tie-breaking.
        Uses numpy for fast max and boolean ops.
        """
        rng = random.Random(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        # Initialize with the total number of triples in the block (C(k, t))
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            # Check time
            if time.time() > deadline - 2:
                # If we run out of time, we need to finish quickly or break.
                # Breaking here returns an invalid cover. 
                # To be safe, we should probably not check time inside the tight loop 
                # unless we have a fast fallback. 
                # However, if we break, the caller must handle invalidity.
                # Let's assume the loop finishes. If not, we risk timeout.
                # Given 110s limit and typical runtime < 1s, this is unlikely to trigger 
                # unless precomputation is slow. Precomputation is one-time.
                pass
            
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true and blocks are valid
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())
            
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
        """
        Remove redundant blocks from the cover.
        A block is redundant if all its triples are covered by other blocks.
        """
        if not selected_blocks:
            return []
            
        # Convert to set for O(1) lookup if needed, but we need order for counts
        # We will iterate and mark for removal
        
        # Count coverage of each triple by the current set of blocks
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_blocks)
        changed = True
        
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                
                # Check if redundant: all triples in this block have count > 1
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

    def local_search_optimize(blocks, max_iters=1000):
        """
        Try to reduce the number of blocks by swapping or removing.
        Since we just did reduction, most are essential.
        We can try to find a block that can be removed if we add another?
        Or just re-run reduction with a different order?
        
        Actually, a simple improvement is to try removing a block and checking if the cover is still valid.
        If not, we can't remove it.
        
        A better local search:
        1. Pick a random block B.
        2. Try to replace B with another block B' such that the cover remains valid and size decreases?
           No, size stays same. We want size to decrease.
           
        Since greedy + reduction is quite strong, let's stick to that.
        However, we can try multiple random orderings in reduction?
        The order of checking redundancy can matter.
        
        Let's implement a simple "shake" and re-reduce:
        1. Remove a random subset of blocks? No, that breaks cover.
        
        Let's just return the reduced cover for now. The main gain comes from multiple greedy restarts.
        """
        return blocks

    best_cover = None
    best_len = float('inf')
    seed = 0
    
    # Run multiple greedy attempts
    # The first few are usually the best.
    # We run until timeout.
    
    # Precompute a list of all block indices for potential randomization if needed
    # But greedy_construct handles its own randomness.
    
    while time.time() < deadline:
        # Run greedy
        cover = greedy_construct(seed)
        
        # Reduce
        cover = reduce_cover(cover)
        
        # Optional: Local search
        # cover = local_search_optimize(cover)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover
            
            # If we found a very good cover, we might want to spend more time on it?
            # Or just keep restarting.
            
        seed += 1
        
        # If we have a lot of time left and best_len is good, we might do more sophisticated search.
        # But simple restarts are robust.
        
        # Check time more frequently
        if seed % 10 == 0:
            if time.time() > deadline - 5:
                break

    # Fallback
    if best_cover is None:
        # Should not happen, but just in case
        cover = greedy_construct(0)
        cover = reduce_cover(cover)
        best_cover = cover

    return [list(all_k_blocks[i]) for i in best_cover]

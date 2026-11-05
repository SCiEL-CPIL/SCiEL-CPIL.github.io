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
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    def reduce_cover(selected_blocks):
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
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search(cover, time_limit):
        """Try 1-swap and 2-for-1 moves to reduce cover size."""
        if not cover:
            return cover
        
        cover_set = set(cover)
        n = len(cover)
        
        # Precompute cover_count for the current cover
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Try 1-swap: remove block i, add block j (j not in cover)
            # A block i can be removed if all its triples are covered by other blocks
            # After removal, some triples may become uncovered (those only covered by i)
            # We need to find a block j that covers all those newly uncovered triples
            
            for i in range(len(cover)):
                if (time.time() - start_time) > time_limit:
                    break
                b_i = cover[i]
                # Check if removing b_i leaves any triple uncovered
                newly_uncovered = []
                for t_idx in block_triples[b_i]:
                    if cover_count[t_idx] == 1:
                        newly_uncovered.append(t_idx)
                
                if not newly_uncovered:
                    # Can simply remove
                    cover.remove(b_i)
                    cover_set.discard(b_i)
                    for t_idx in block_triples[b_i]:
                        cover_count[t_idx] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1: remove b_i, add b_j and b_k
            # Only try if cover is small enough
            if len(cover) > 100:
                break
            
            for i in range(len(cover)):
                if (time.time() - start_time) > time_limit:
                    break
                b_i = cover[i]
                newly_uncovered = []
                for t_idx in block_triples[b_i]:
                    if cover_count[t_idx] == 1:
                        newly_uncovered.append(t_idx)
                
                if not newly_uncovered:
                    continue  # Already handled above
                
                # Find blocks that cover all newly_uncovered triples
                # This is expensive, so limit candidates
                candidate_blocks = []
                for t_idx in newly_uncovered:
                    candidate_blocks.extend(triple_blocks[t_idx])
                
                # Deduplicate
                candidate_blocks = list(set(candidate_blocks))
                # Remove blocks already in cover (except b_i)
                candidate_blocks = [b for b in candidate_blocks if b not in cover_set or b == b_i]
                
                # Limit candidates
                random.shuffle(candidate_blocks)
                candidate_blocks = candidate_blocks[:50]
                
                for j in candidate_blocks:
                    if j == b_i:
                        continue
                    # Check if j covers all newly_uncovered
                    j_triples = set(block_triples[j])
                    if not all(t in j_triples for t in newly_uncovered):
                        continue
                    
                    # Try adding just j
                    # Check if removing i and adding j is valid
                    # We need all triples that were only covered by i to be covered by j
                    # Which we already checked
                    cover[i] = j
                    cover_set.discard(b_i)
                    cover_set.add(j)
                    # Update cover_count
                    for t_idx in block_triples[b_i]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_triples[j]:
                        cover_count[t_idx] += 1
                    improved = True
                    break
                
                if improved:
                    break
        
        return cover

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
            best_cover = list(cover)
        
        seed += 1
    
    # Final local search on best cover
    if best_cover is not None:
        time_remaining = timeout - (time.time() - start_time)
        if time_remaining > 5:
            best_cover = local_search(best_cover, start_time + timeout - 2)
    
    if best_cover is None:
        cover = greedy_randomized(0)
        return [list(blocks[b_idx]) for b_idx in cover]
        
    return [list(blocks[b_idx]) for b_idx in best_cover]

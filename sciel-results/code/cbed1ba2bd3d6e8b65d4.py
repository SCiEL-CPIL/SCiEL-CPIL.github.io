import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using pre-computed blocks, 
    vectorized greedy, and local search with cover counts.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map each block to the list of triple indices it covers
    # block_triples[i] is a list of triple indices covered by block i
    block_triples = []
    for b in all_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(np.array(tids, dtype=np.int32))

    # Map each triple to the list of block indices that cover it
    # This is useful for the "re-cover" step in local search
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)
    
    # Convert triple_blocks to numpy arrays for faster set operations if needed, 
    # but lists are often faster for small intersections. 
    # We'll keep them as lists for now, or convert to sets if intersections are frequent.
    triple_blocks_sets = [set(lst) for lst in triple_blocks]

    def greedy_cover(seed, time_budget):
        """
        Greedy selection of blocks to cover all triples.
        Uses a gain vector to track how many uncovered triples each block covers.
        """
        rng = random.Random(seed)
        # gain[b] = number of currently uncovered triples covered by block b
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        selected = []

        while is_uncovered.any():
            if time.time() > time_budget:
                break
            
            # Find blocks with max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Get indices of blocks with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())
            
            selected.append(b_idx)
            
            # Update gain for all blocks
            # For each triple covered by b_idx, if it was uncovered, 
            # decrement gain for all blocks that also cover that triple.
            tids = block_triples[b_idx]
            for t_idx in tids:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # All blocks covering this triple lose 1 gain
                    # We need to iterate over blocks covering t_idx
                    # This is the expensive part. 
                    # Optimization: Only update gain if we are maintaining a global gain array.
                    # Since we pick max gain, we need accurate gains.
                    for b_cand in triple_blocks[t_idx]:
                        gain[b_cand] -= 1

        return selected

    def complete_cover(selected, deadline):
        """
        If greedy broke early, finish covering.
        """
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1
        
        while True:
            uncovered_mask = cover_count == 0
            if not uncovered_mask.any():
                break
            if time.time() > deadline:
                break
            
            # Find block covering most uncovered triples
            # Sample blocks to speed up
            num_samples = min(num_blocks, 5000)
            samples = random.sample(range(num_blocks), num_samples)
            
            best_b = -1
            best_gain = 0
            for b_idx in samples:
                tids = block_triples[b_idx]
                # Count how many of these tids are currently uncovered
                cnt = np.sum(uncovered_mask[tids])
                if cnt > best_gain:
                    best_gain = cnt
                    best_b = b_idx
            
            if best_b == -1:
                # Fallback: exhaustive search (slow, but safe)
                for b_idx in range(num_blocks):
                    tids = block_triples[b_idx]
                    cnt = np.sum(uncovered_mask[tids])
                    if cnt > best_gain:
                        best_gain = cnt
                        best_b = b_idx
                if best_b == -1:
                    break
            
            selected.append(best_b)
            cover_count[block_triples[best_b]] += 1

        return selected

    def reduce_cover(selected):
        """
        Remove redundant blocks.
        A block is redundant if every triple it covers is covered by at least one other block.
        """
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1

        active = [True] * len(selected)
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                b_idx = selected[i]
                tids = block_triples[b_idx]
                # If min count > 1, all triples in this block are covered by others
                if np.min(cover_count[tids]) > 1:
                    active[i] = False
                    cover_count[tids] -= 1
                    changed = True
        
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search(selected, deadline):
        """
        Try to remove a block and re-cover the gap with 1 block (swap) or 0 blocks (redundant).
        """
        if len(selected) < 2:
            return selected
            
        selected = list(selected)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1

        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            order = list(range(len(selected)))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 5:
                    break
                
                b_idx = selected[i]
                # Temporarily remove
                cover_count[block_triples[b_idx]] -= 1
                uncovered_mask = cover_count == 0
                
                if not uncovered_mask.any():
                    # Redundant
                    selected.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all uncovered triples
                unc_idx = np.where(uncovered_mask)[0]
                if len(unc_idx) <= 35: # Max triples in a block
                    # Find intersection of blocks covering each uncovered triple
                    # Start with blocks covering the first uncovered triple
                    candidates = set(triple_blocks[unc_idx[0]])
                    for t in unc_idx[1:]:
                        candidates.intersection_update(triple_blocks[t])
                        if not candidates:
                            break
                    
                    # Exclude blocks already in selected (except the one we removed? 
                    # Actually, if we removed b_idx, it's not in selected anymore.
                    # But we might have added it back? No, we popped it? 
                    # Wait, we haven't popped it yet. We just decremented count.
                    # So b_idx is still in 'selected' list.
                    # We should exclude blocks in 'selected' to avoid swapping b for b.
                    already = set(selected)
                    candidates -= already
                    
                    if candidates:
                        new_b = next(iter(candidates))
                        # Perform swap
                        selected.pop(i)
                        selected.append(new_b)
                        cover_count[block_triples[new_b]] += 1
                        improved = True
                        break
                    else:
                        # Revert
                        cover_count[block_triples[b_idx]] += 1
                else:
                    # Too many uncovered, revert
                    cover_count[block_triples[b_idx]] += 1

        return reduce_cover(selected)

    best_cover = None

    # Phase 1: Multiple greedy restarts
    seed = 0
    while time.time() < deadline - 20:
        budget = deadline - 15
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 10)
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            # If we hit a very good score, we can tighten the deadline for local search
            # but for now, keep it simple.
        
        seed += 1

    if best_cover is None:
        best_cover = greedy_cover(0, deadline - 5)
        best_cover = complete_cover(best_cover, deadline - 2)
        best_cover = reduce_cover(best_cover)

    # Phase 2: Local search on the best cover
    best_cover = local_search(best_cover, deadline - 3)

    # Phase 3: If time remains, try more restarts
    while time.time() < deadline - 3:
        seed = random.randint(1000, 100000)
        budget = deadline - 5
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 3)
        cover = reduce_cover(cover)
        if len(cover) < len(best_cover):
            best_cover = cover
            # Apply local search immediately to the new best
            best_cover = local_search(best_cover, deadline - 3)

    # Final safety: ensure validity
    cover_count = np.zeros(num_triples, dtype=np.int32)
    for b_idx in best_cover:
        cover_count[block_triples[b_idx]] += 1
    if (cover_count == 0).any():
        # If invalid, force completion
        best_cover = complete_cover(best_cover, time.time() + 10)
        best_cover = reduce_cover(best_cover)

    # Convert indices back to lists of points
    return [list(all_blocks[b_idx]) for b_idx in best_cover]

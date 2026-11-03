import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for safety

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast counting
    t_subset_to_idx = {frozenset(s): i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    def get_idx(s):
        return t_subset_to_idx[frozenset(s)]

    def _greedy_cover(v, k, t, rng):
        """Fast greedy construction to get a valid starting covering."""
        uncovered = set(range(num_t_subsets))
        blocks = []
        block_set = set()
        
        while uncovered:
            # Pick a random uncovered t-subset as seed
            seed_idx = rng.choice(tuple(uncovered))
            seed = list(all_t_subsets[seed_idx])
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = t_subset_to_idx.get(fs)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in block_set:
                blocks.append(b_tuple)
                block_set.add(b_tuple)
                for c in itertools.combinations(block, t):
                    idx = t_subset_to_idx[frozenset(c)]
                    uncovered.discard(idx)
        
        return blocks

    # Initialize
    blocks = _greedy_cover(v, k, t, rng)
    block_set = set(blocks)
    
    # Count occurrences of each t-subset
    counts = [0] * num_t_subsets
    for b in blocks:
        for c in itertools.combinations(b, t):
            idx = t_subset_to_idx[frozenset(c)]
            counts[idx] += 1
            
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def update_counts(block, sign):
        """Update counts array for adding or removing a block."""
        for c in itertools.combinations(block, t):
            idx = t_subset_to_idx[frozenset(c)]
            counts[idx] += sign

    no_improve_iter = 0
    max_no_improve = 100 # Lower threshold to allow more restarts
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to remove redundant blocks
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for c in itertools.combinations(b, t):
                idx = t_subset_to_idx[frozenset(c)]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                update_counts(b, -1)
                blocks.pop(i)
                block_set.discard(b)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                improved = True
                break # Restart scan after a removal
        
        if improved:
            no_improve_iter = 0
            continue
            
        # 2. Try 2-for-1 swaps
        # For each pair of blocks, check if their union of critical triples can be covered by one block
        found_swap = False
        # Limit pairs to avoid O(N^2) explosion if N is large, but N~70 is fine
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_blocks):
                bi = blocks[i]
                bj = blocks[j]
                
                # Find triples covered ONLY by bi or bj
                # A triple is critical if count == 1 and it is in bi or bj
                critical_triples = []
                points_union = set()
                
                # Check triples in bi
                for c in itertools.combinations(bi, t):
                    idx = t_subset_to_idx[frozenset(c)]
                    if counts[idx] == 1:
                        critical_triples.append(c)
                        points_union.update(c)
                        
                # Check triples in bj
                for c in itertools.combinations(bj, t):
                    idx = t_subset_to_idx[frozenset(c)]
                    if counts[idx] == 1:
                        # Ensure it's not already added (if bi and bj share a critical triple, count would be 2, so they can't share a critical triple with count 1)
                        critical_triples.append(c)
                        points_union.update(c)
                        
                if len(points_union) <= k:
                    # We can form a block containing points_union
                    # Pad with random points to reach size k
                    new_block_set = set(points_union)
                    cands = [x for x in range(v) if x not in new_block_set]
                    rng.shuffle(cands)
                    for x in cands:
                        if len(new_block_set) >= k:
                            break
                        new_block_set.add(x)
                        
                    new_block = tuple(sorted(new_block_set))
                    
                    if new_block in block_set:
                        continue
                    if new_block == bi or new_block == bj:
                        continue
                        
                    # Verify that this new block actually covers all critical triples
                    # It should, by construction, as it contains points_union
                    # But let's be safe: check if all critical triples are subsets of new_block
                    valid_swap = True
                    nb_set = set(new_block)
                    for c in critical_triples:
                        if not set(c).issubset(nb_set):
                            valid_swap = False
                            break
                            
                    if valid_swap:
                        # Perform swap
                        update_counts(bi, -1)
                        update_counts(bj, -1)
                        update_counts(new_block, 1)
                        
                        # Remove bi, bj, add new_block
                        # Order matters for indices
                        if i < j:
                            blocks.pop(j)
                            blocks.pop(i)
                        else:
                            blocks.pop(i)
                            blocks.pop(j)
                        blocks.append(new_block)
                        
                        block_set.discard(bi)
                        block_set.discard(bj)
                        block_set.add(new_block)
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = list(blocks)
                        
                        found_swap = True
                        break # Restart pair search after a swap
            if found_swap:
                break
                
        if found_swap:
            no_improve_iter = 0
            continue
            
        # 3. Perturbation / Restart if stuck
        no_improve_iter += 1
        if no_improve_iter > max_no_improve:
            # Remove a random block and repair
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_set = set(block_set)
                old_counts = list(counts)
                
                i = rng.randrange(len(blocks))
                b = blocks[i]
                update_counts(b, -1)
                blocks.pop(i)
                block_set.discard(b)
                
                # Find uncovered triples
                unc_list = [idx for idx in range(num_t_subsets) if counts[idx] == 0]
                
                # Greedily add blocks to cover uncovered
                while unc_list:
                    seed_idx = rng.choice(unc_list)
                    seed = list(all_t_subsets[seed_idx])
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t - 1):
                                fs = frozenset(list(rest) + [x])
                                idx = t_subset_to_idx.get(fs)
                                if idx is not None and counts[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                        
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in block_set:
                        blocks.append(new_b_tuple)
                        block_set.add(new_b_tuple)
                        update_counts(new_b_tuple, 1)
                        
                        # Update unc_list
                        unc_list = [idx for idx in range(num_t_subsets) if counts[idx] == 0]
                        
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    
                no_improve_iter = 0
            else:
                no_improve_iter += 1
                
    return [list(b) for b in best_blocks]

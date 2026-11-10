import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precomputation
    # 1. Map triplets to indices
    triplets = list(itertools.combinations(range(v), t))
    triplet_index = {trip: i for i, trip in enumerate(triplets)}
    n_triplets = len(triplets)
    
    # 2. Generate all blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Precompute block_masks as list of ints
    block_masks = [0] * n_blocks
    # Precompute triple_to_blocks
    triple_to_blocks = [[] for _ in range(n_triplets)]
    
    for j, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triplet_index[triple]
            mask |= (1 << idx)
            triple_to_blocks[idx].append(j)
        block_masks[j] = mask
    
    def reduce_cover(selected_indices):
        """Remove redundant blocks."""
        if not selected_indices:
            return selected_indices
        
        # Compute coverage count for each triple
        coverage_count = [0] * n_triplets
        for j in selected_indices:
            for triple in itertools.combinations(blocks[j], t):
                idx = triplet_index[triple]
                coverage_count[idx] += 1
        
        remaining = list(selected_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(remaining)):
                j = remaining[i]
                can_remove = True
                for triple in itertools.combinations(blocks[j], t):
                    idx = triplet_index[triple]
                    if coverage_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for triple in itertools.combinations(blocks[j], t):
                        idx = triplet_index[triple]
                        coverage_count[idx] -= 1
                    remaining.pop(i)
                    changed = True
                    break
        return remaining

    def fast_polish(selected_indices, time_limit):
        """Try 1-for-1 swaps using triple_to_blocks for candidate generation."""
        if not selected_indices:
            return selected_indices
        
        selected_set = set(selected_indices)
        best_size = len(selected_indices)
        best_set = set(selected_indices)
        
        # Compute initial coverage counts
        def get_counts(s):
            counts = [0] * n_triplets
            for j in s:
                for triple in itertools.combinations(blocks[j], t):
                    idx = triplet_index[triple]
                    counts[idx] += 1
            return counts
        
        current_counts = get_counts(selected_set)
        
        while time.time() < time_limit:
            # Find a block j that is not redundant (has only_j)
            # Or just pick a random block
            if not selected_set:
                break
            j = random.choice(list(selected_set))
            
            # Find only_j using current_counts
            only_j = []
            for triple in itertools.combinations(blocks[j], t):
                idx = triplet_index[triple]
                if current_counts[idx] == 1:
                    only_j.append(idx)
            
            if not only_j:
                # j is redundant
                selected_set.remove(j)
                # Update counts
                for triple in itertools.combinations(blocks[j], t):
                    idx = triplet_index[triple]
                    current_counts[idx] -= 1
                best_size = len(selected_set)
                best_set = set(selected_set)
                continue
            
            # Find candidates covering only_j
            candidates = set(triple_to_blocks[only_j[0]])
            for idx in only_j[1:]:
                candidates.intersection_update(triple_to_blocks[idx])
            
            candidates.discard(j)
            
            if not candidates:
                continue
            
            c = random.choice(list(candidates))
            
            # Swap j with c
            # Remove j, add c
            # Update counts
            for triple in itertools.combinations(blocks[j], t):
                idx = triplet_index[triple]
                current_counts[idx] -= 1
            for triple in itertools.combinations(blocks[c], t):
                idx = triplet_index[triple]
                current_counts[idx] += 1
                
            selected_set.remove(j)
            selected_set.add(c)
            
            # Try to reduce
            # We can run a quick reduction on the new set
            # But updating counts is already done.
            # Just check for redundancies
            changed = True
            while changed:
                changed = False
                # Randomly pick a block to check? Or iterate all?
                # Iterate all is O(M*K). M is ~76. Fast.
                for jj in list(selected_set):
                    redundant = True
                    for triple in itertools.combinations(blocks[jj], t):
                        idx = triplet_index[triple]
                        if current_counts[idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove jj
                        for triple in itertools.combinations(blocks[jj], t):
                            idx = triplet_index[triple]
                            current_counts[idx] -= 1
                        selected_set.remove(jj)
                        changed = True
                        break
            
            if len(selected_set) < best_size:
                best_size = len(selected_set)
                best_set = set(selected_set)
            else:
                # Revert swap
                selected_set.remove(c)
                selected_set.add(j)
                # Revert counts
                for triple in itertools.combinations(blocks[c], t):
                    idx = triplet_index[triple]
                    current_counts[idx] -= 1
                for triple in itertools.combinations(blocks[j], t):
                    idx = triplet_index[triple]
                    current_counts[idx] += 1

        return list(best_set)

    # Main Loop
    best_covering = None
    best_size = float('inf')
    
    start_time = time.time()
    max_time = 110
    
    # Number of restarts
    while time.time() - start_time < max_time:
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Greedy selection
            for j in range(n_blocks):
                inter = block_masks[j] & uncovered
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [j]
                    elif gain == best_gain:
                        best_candidates.append(j)
                
            if best_gain == 0:
                break
                
            chosen_j = random.choice(best_candidates)
            selected_indices.append(chosen_j)
            uncovered &= ~block_masks[chosen_j]
            
            if time.time() - start_time > max_time:
                break
        
        if time.time() - start_time > max_time:
            break
            
        # Polish
        reduced = reduce_cover(selected_indices)
        # Allocate remaining time for polish
        polish_time = min(time.time() + 10, start_time + max_time) # Give up to 10s per restart?
        # Better: just give a small chunk
        polish_time = time.time() + 5
        
        polished = fast_polish(reduced, polish_time)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_covering = polished
            
        if best_size <= 69:
            break

    if best_covering is None:
        return []
        
    return [list(blocks[i]) for i in best_covering]

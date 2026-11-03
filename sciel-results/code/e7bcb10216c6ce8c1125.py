import itertools
import time
import random
import sys
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110

    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute masks and subset indices for each block
    block_masks = []
    block_subset_indices = []
    
    # Also build reverse index: for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(num_t_subsets)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
            tsub_blocks[idx].append(b_idx)
        block_masks.append(mask)
        block_subset_indices.append(subs_idx)
        
    def greedy_repair(uncovered_mask, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks)
            
        if not uncovered_mask:
            return existing_blocks

        # Use a heap to speed up finding the best block
        # Heap elements: (-gain, block_idx)
        heap = []
        # Initialize heap with all blocks
        for i in range(num_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > 0:
                heapq.heappush(heap, (-gain, i))
        
        while uncovered_mask and heap:
            # Pop the best block
            neg_gain, block_idx = heapq.heappop(heap)
            gain = -neg_gain
            
            # Check if this gain is still valid
            actual_gain = (uncovered_mask & block_masks[block_idx]).bit_count()
            
            if actual_gain != gain:
                # Stale entry, push updated entry if still useful
                if actual_gain > 0:
                    heapq.heappush(heap, (-actual_gain, block_idx))
                continue
            
            if actual_gain == 0:
                continue
                
            # Select this block
            existing_blocks.append(block_idx)
            uncovered_mask &= ~block_masks[block_idx]
            
        return existing_blocks

    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            for s_idx in subs:
                cover_count[s_idx] -= 1
                
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                pass
            else:
                for s_idx in subs:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions using reverse index for candidate generation.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * num_t_subsets
        for b in block_set:
            for ts_idx in block_subset_indices[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_subset_indices[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_subset_indices[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_subset_indices[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_subset_indices[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates using set intersection via reverse index
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_subset_indices[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_subset_indices[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    full_mask = (1 << num_t_subsets) - 1
    initial_blocks = greedy_repair(full_mask)
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Local search on initial solution
    current_blocks = local_search(current_blocks, 42)
    current_blocks = reduce_blocks(current_blocks)
    if len(current_blocks) < best_size:
        best_blocks = current_blocks
        best_size = len(best_blocks)
    
    while time.time() < deadline:
        if not current_blocks:
            break
            
        perturbed_blocks = list(current_blocks)
        remove_idx = random.randint(0, len(perturbed_blocks) - 1)
        perturbed_blocks.pop(remove_idx)
        
        covered_mask = 0
        for idx in perturbed_blocks:
            covered_mask |= block_masks[idx]
            
        uncovered_mask = full_mask & ~covered_mask
        
        repaired_blocks = greedy_repair(uncovered_mask, perturbed_blocks)
        new_blocks = reduce_blocks(repaired_blocks)
        
        # Apply local search to the new solution
        new_blocks = local_search(new_blocks, random.randint(0, 10**6))
        new_blocks = reduce_blocks(new_blocks)
        
        new_size = len(new_blocks)
        
        if new_size < best_size:
            best_blocks = new_blocks
            best_size = new_size
            current_blocks = new_blocks
        else:
            if new_size == best_size:
                if random.random() < 0.5:
                    current_blocks = new_blocks
            else:
                if random.random() < 0.2:
                    current_blocks = new_blocks
                else:
                    current_blocks = best_blocks

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks

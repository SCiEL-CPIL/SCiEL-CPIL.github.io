import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, leave buffer)
    
    # Precompute all t-subsets and map to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # For each block, compute the bitmask of triples it covers
    block_masks = [0] * num_blocks
    # For each triple, store list of block indices containing it
    blocks_for_triple = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_for_triple[idx].append(i)
        block_masks[i] = mask
    
    def greedy_construct():
        """Greedy set cover with better tie-breaking."""
        uncovered_mask = (1 << num_triples) - 1
        selected = set()
        
        while uncovered_mask:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_gain = 0
            best_total = 0  # tie-breaker: prefer blocks with more total coverage
            
            for i in range(num_blocks):
                if i in selected:
                    continue
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain or (gain == best_gain and gain > 0):
                    total = block_masks[i].bit_count()
                    if gain > best_gain or total > best_total:
                        best_gain = gain
                        best_total = total
                        best_idx = i
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        return selected
    
    def get_unique_triple_indices(block_idx, selected_set):
        """Returns list of triple indices covered ONLY by block_idx among selected_set."""
        unique = []
        block_tups = blocks[block_idx]
        for triple in itertools.combinations(block_tups, t):
            idx = triple_to_idx[triple]
            # Check if any OTHER selected block covers this triple
            covered_by_other = False
            for other_idx in blocks_for_triple[idx]:
                if other_idx != block_idx and other_idx in selected_set:
                    covered_by_other = True
                    break
            if not covered_by_other:
                unique.append(idx)
        return unique
    
    def try_remove_redundant(selected_set):
        """Remove any blocks that are redundant."""
        removed = False
        for b_idx in list(selected_set):
            if time.time() > deadline - 5:
                break
            unique_triples = get_unique_triple_indices(b_idx, selected_set)
            if not unique_triples:
                selected_set.remove(b_idx)
                removed = True
        return removed
    
    def find_swap_candidates(unique_triple_indices, selected_set, exclude_idx):
        """Find blocks that cover all unique triples using intersection of blocks_for_triple."""
        if not unique_triple_indices:
            return []
        
        candidates = set(blocks_for_triple[unique_triple_indices[0]])
        for ut in unique_triple_indices[1:]:
            candidates &= set(blocks_for_triple[ut])
            if not candidates:
                return []
        
        candidates.discard(exclude_idx)
        # Remove blocks already in selected_set (they can't help as swaps)
        candidates -= selected_set
        return list(candidates)
    
    def score_candidate(c_idx, selected_set, exclude_idx):
        """Score a swap candidate: prefer blocks that create more redundancy."""
        # Count how many of c_idx's triples are already covered by other selected blocks
        other_union_mask = 0
        for other in selected_set:
            if other != exclude_idx:
                other_union_mask |= block_masks[other]
        return (block_masks[c_idx] & other_union_mask).bit_count()
    
    def try_1swap(selected_set):
        """Try to replace one block with another that covers its unique triples."""
        improved = False
        
        for b_idx in list(selected_set):
            if time.time() > deadline - 5:
                return improved
            
            unique_triples = get_unique_triple_indices(b_idx, selected_set)
            
            if not unique_triples:
                selected_set.remove(b_idx)
                improved = True
                continue
            
            candidates = find_swap_candidates(unique_triples, selected_set, b_idx)
            
            if not candidates:
                continue
            
            # Score candidates
            best_c = -1
            best_score = -1
            for c_idx in candidates:
                score = score_candidate(c_idx, selected_set, b_idx)
                if score > best_score:
                    best_score = score
                    best_c = c_idx
            
            if best_c != -1:
                selected_set.remove(b_idx)
                selected_set.add(best_c)
                improved = True
                break  # Restart scan after a swap
        
        return improved
    
    def try_2swap(selected_set):
        """Try to replace two blocks with one or two different blocks."""
        if time.time() > deadline - 10:
            return False
        
        improved = False
        block_list = list(selected_set)
        n = len(block_list)
        
        if n < 2:
            return False
        
        # Try removing two blocks and seeing if we can cover their unique triples with one or two new blocks
        for i in range(n):
            if time.time() > deadline - 5:
                return improved
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return improved
                
                b1 = block_list[i]
                b2 = block_list[j]
                
                # Find unique triples of b1 and b2 (considering all other selected blocks)
                others = selected_set - {b1, b2}
                
                unique1 = []
                for triple in itertools.combinations(blocks[b1], t):
                    idx = triple_to_idx[triple]
                    covered = any(o in blocks_for_triple[idx] for o in others)
                    if not covered:
                        unique1.append(idx)
                
                unique2 = []
                for triple in itertools.combinations(blocks[b2], t):
                    idx = triple_to_idx[triple]
                    covered = any(o in blocks_for_triple[idx] for o in others)
                    if not covered:
                        unique2.append(idx)
                
                if not unique1 and not unique2:
                    # Both are redundant
                    selected_set.discard(b1)
                    selected_set.discard(b2)
                    improved = True
                    return True
                
                all_unique = set(unique1) | set(unique2)
                
                # Try to cover all_unique with a single block
                if all_unique:
                    candidates = set(blocks_for_triple[list(all_unique)[0]])
                    for ut in list(all_unique)[1:]:
                        candidates &= set(blocks_for_triple[ut])
                        if not candidates:
                            break
                    
                    candidates.discard(b1)
                    candidates.discard(b2)
                    candidates -= others
                    
                    if candidates:
                        # Pick best candidate
                        best_c = -1
                        best_score = -1
                        other_union = 0
                        for o in others:
                            other_union |= block_masks[o]
                        for c_idx in candidates:
                            score = (block_masks[c_idx] & other_union).bit_count()
                            if score > best_score:
                                best_score = score
                                best_c = c_idx
                        
                        if best_c != -1:
                            selected_set.discard(b1)
                            selected_set.discard(b2)
                            selected_set.add(best_c)
                            improved = True
                            return True
        
        return improved
    
    def local_search(selected_set, max_rounds=30):
        """Iterative local search."""
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            made_progress = False
            
            if try_remove_redundant(selected_set):
                made_progress = True
            
            if try_1swap(selected_set):
                made_progress = True
            
            if try_2swap(selected_set):
                made_progress = True
            
            if made_progress and try_remove_redundant(selected_set):
                pass
            
            if not made_progress:
                break
        
        return selected_set
    
    # Multiple restarts
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        if r == 0:
            current_set = greedy_construct()
        else:
            if best_solution is not None:
                current_set = set(best_solution)
                # Perturb: remove 2-5 random blocks
                n_remove = min(2 + r // 2, max(1, len(current_set) - 1))
                to_remove = random.sample(list(current_set), n_remove)
                for b in to_remove:
                    current_set.discard(b)
                
                # Re-greedy to restore coverage
                uncovered = (1 << num_triples) - 1
                for b in current_set:
                    uncovered &= ~block_masks[b]
                
                while uncovered:
                    if time.time() > deadline - 10:
                        break
                    best_idx = -1
                    best_gain = 0
                    best_total = 0
                    for i in range(num_blocks):
                        if i in current_set:
                            continue
                        gain = (block_masks[i] & uncovered).bit_count()
                        if gain > best_gain or (gain == best_gain and gain > 0):
                            total = block_masks[i].bit_count()
                            if gain > best_gain or total > best_total:
                                best_gain = gain
                                best_total = total
                                best_idx = i
                    if best_idx == -1:
                        break
                    current_set.add(best_idx)
                    uncovered &= ~block_masks[best_idx]
            else:
                current_set = greedy_construct()
        
        current_set = local_search(current_set)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_solution = set(current_set)
    
    # Final polish on best
    if best_solution is not None:
        best_solution = local_search(best_solution, max_rounds=50)
    
    final_blocks = [blocks[i] for i in best_solution]
    
    # Ensure distinctness
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(list(b))
    
    return unique_blocks

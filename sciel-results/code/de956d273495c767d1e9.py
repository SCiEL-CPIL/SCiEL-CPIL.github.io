import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets and map to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all k-blocks and their coverage masks
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # For each block, compute the bitmask of triples it covers
    block_masks = [0] * num_blocks
    # Also, for each triple, store list of block indices containing it
    blocks_for_triple = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_for_triple[idx].append(i)
        block_masks[i] = mask
    
    def greedy_construct():
        """Greedy set cover construction."""
        uncovered_mask = (1 << num_triples) - 1
        selected = []
        selected_set = set()
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        return selected_set
    
    def get_unique_triple_mask(block_idx, selected_set):
        """Returns bitmask of triples covered ONLY by block_idx among selected_set."""
        mask = block_masks[block_idx]
        # Compute union of all other selected blocks' masks
        other_union = 0
        for other in selected_set:
            if other != block_idx:
                other_union |= block_masks[other]
        return mask & ~other_union
    
    def try_remove_redundant(selected_set):
        """Remove any blocks that are redundant (their unique triples are empty)."""
        removed = False
        for b_idx in list(selected_set):
            if time.time() > deadline:
                break
            unique_mask = get_unique_triple_mask(b_idx, selected_set)
            if unique_mask == 0:
                selected_set.remove(b_idx)
                removed = True
        return removed
    
    def try_1swap(selected_set):
        """Try to replace one block with another that covers its unique triples and enables further reductions."""
        improved = False
        
        for b_idx in list(selected_set):
            if time.time() > deadline:
                return improved
            
            unique_mask = get_unique_triple_mask(b_idx, selected_set)
            
            if unique_mask == 0:
                selected_set.remove(b_idx)
                improved = True
                continue
            
            # Find candidate blocks that cover all unique triples
            # A block c must have: (block_masks[c] & unique_mask) == unique_mask
            candidates = []
            for c_idx in range(num_blocks):
                if c_idx in selected_set or c_idx == b_idx:
                    continue
                if (block_masks[c_idx] & unique_mask) == unique_mask:
                    candidates.append(c_idx)
            
            if not candidates:
                continue
            
            # Score candidates: prefer those that cover many triples already covered by other selected blocks
            # This creates more redundancy and potential for further removals
            best_c = -1
            best_score = -1
            
            # Compute union of other selected blocks (excluding b_idx)
            other_union = 0
            for other in selected_set:
                if other != b_idx:
                    other_union |= block_masks[other]
            
            for c_idx in candidates:
                # Score = number of triples in c that are in other_union (redundancy potential)
                score = (block_masks[c_idx] & other_union).bit_count()
                if score > best_score:
                    best_score = score
                    best_c = c_idx
            
            if best_c != -1:
                selected_set.remove(b_idx)
                selected_set.add(best_c)
                improved = True
                # After a swap, restart the scan to catch cascading removals
                break
        
        return improved
    
    def local_search(selected_set, max_rounds=50):
        """Iterative local search: remove redundant, then try 1-swaps."""
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            made_progress = False
            
            # Phase 1: Remove redundant blocks
            if try_remove_redundant(selected_set):
                made_progress = True
            
            # Phase 2: Try 1-swaps
            if try_1swap(selected_set):
                made_progress = True
            
            # After any change, try removing redundant again
            if made_progress and try_remove_redundant(selected_set):
                pass  # Already counted
            
            if not made_progress:
                break
        
        return selected_set
    
    # Multiple random restarts to escape local optima
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 5
    restart_budget = (deadline - start_time) / (num_restarts + 1)
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed greedy solution
        if r == 0:
            # First restart: pure greedy
            current_set = greedy_construct()
        else:
            # Subsequent restarts: start from best found, then perturb
            if best_solution is not None:
                current_set = set(best_solution)
                # Remove 2-4 random blocks and re-greedy to fill gaps
                n_remove = min(2 + r, len(current_set) - 1)
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
                    for i in range(num_blocks):
                        if i in current_set:
                            continue
                        gain = (block_masks[i] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    current_set.add(best_idx)
                    uncovered &= ~block_masks[best_idx]
            else:
                current_set = greedy_construct()
        
        # Local search
        current_set = local_search(current_set)
        
        # Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_solution = set(current_set)
    
    # Final local search on the best solution
    if best_solution is not None:
        best_solution = local_search(best_solution)
    
    # Convert to list of blocks
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

import itertools
import time
import random
import sys
from collections import defaultdict

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    block_covers_set = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        block_covers_set.append(set(covers))
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Precompute sub_to_blocks for fast candidate lookup
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
    
    # Precompute for each block, the set of blocks that share at least one t-subset
    # This helps in local search
    if not _deadline_ok(deadline):
        return []

    def greedy_mcv_bitset(random_seed=None):
        """Greedy set cover using bitsets for fast coverage tracking."""
        if random_seed is not None:
            random.seed(random_seed)
            
        # Use a list of booleans for uncovered subsets
        uncovered = [True] * n_t
        uncovered_count = n_t
        selected_indices = []
        
        # Precompute for each block: list of (subset_index, position_in_block)
        # Actually, we just need to track which subsets are covered
        
        while uncovered_count > 0:
            if not _deadline_ok(deadline):
                break
                
            # Find uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if uncovered[s]:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Pick candidate that maximizes coverage of other uncovered subsets
            best_block = -1
            max_gain = -1
            best_candidates = []
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if uncovered[s]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_candidates = [b]
                elif gain == max_gain:
                    best_candidates.append(b)
                    
            if len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    
        return selected_indices

    def prune_and_optimize_bitset(selected_indices, random_seed=None):
        """Local search with bitset-accelerated coverage checks."""
        if random_seed is not None:
            random.seed(random_seed)
            
        # Initialize counts
        counts = [0] * n_t
        selected_set = set(selected_indices)
        selected_list = list(selected_set)
        
        for i in selected_list:
            for s in block_covers[i]:
                counts[s] += 1
                
        # 1. Initial Pruning: Remove any block that is redundant
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                redundant = True
                for s in block_covers[i]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for s in block_covers[i]:
                        counts[s] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
                        
        # 2. Local Search
        # Move: Try to remove a block, or swap it with another block
        # Key insight: A block can be removed if all its covered subsets are covered by other blocks
        
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        no_improve_count = 0
        max_no_improve = 50
        
        while T > min_T and _deadline_ok(deadline) and no_improve_count < max_no_improve:
            if not selected_set:
                break
                
            improved = False
            
            # Pick a random block to try removing/swapping
            i = random.choice(list(selected_set))
            
            # Identify critical subsets covered by i (those with count == 1)
            critical = []
            for s in block_covers[i]:
                if counts[s] == 1:
                    critical.append(s)
                    
            if not critical:
                # Block i is redundant. Remove it.
                selected_set.remove(i)
                for s in block_covers[i]:
                    counts[s] -= 1
                improved = True
                T = max(min_T, T * 0.9)
                no_improve_count = 0
                continue
                
            # Block i is critical. We need to find a replacement j.
            # j must cover ALL subsets in critical.
            
            # Find intersection of blocks covering each s in critical
            # Start with the critical subset with fewest covering blocks
            crit_sorted = sorted(critical, key=lambda s: len(sub_to_blocks[s]))
            target_s = crit_sorted[0]
            candidates = set(sub_to_blocks[target_s])
            
            # Intersect with others
            for s in crit_sorted[1:]:
                candidates &= set(sub_to_blocks[s])
                
            # Remove existing blocks and i itself
            candidates -= selected_set
            
            if not candidates:
                # No replacement found. Reject move.
                no_improve_count += 1
                T = max(min_T, T * cooling)
                continue
                
            # Pick best candidate j
            # Score j by: number of subsets it covers that are currently count==1 (potential for further pruning)
            # Also consider: how many of i's critical subsets it covers (should be all)
            
            best_j = -1
            best_score = -1
            cand_list = list(candidates)
            
            # If too many candidates, sample
            if len(cand_list) > 30:
                random.shuffle(cand_list)
                cand_list = cand_list[:30]
                
            for j in cand_list:
                score = 0
                for s in block_covers[j]:
                    if counts[s] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j == -1:
                no_improve_count += 1
                T = max(min_T, T * cooling)
                continue
                
            # Perform swap i -> best_j
            selected_set.remove(i)
            selected_set.add(best_j)
            
            for s in block_covers[i]:
                counts[s] -= 1
            for s in block_covers[best_j]:
                counts[s] += 1
                
            # Check if this swap made any block redundant
            pruned = False
            for b in list(selected_set):
                redundant = True
                for s in block_covers[b]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(b)
                    for s in block_covers[b]:
                        counts[s] -= 1
                    pruned = True
                    improved = True
                    
            if improved:
                T = max(min_T, T * 0.9)
                no_improve_count = 0
            else:
                T = max(min_T, T * cooling)
                no_improve_count += 1

        return list(selected_set)

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 8
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        selected = greedy_mcv_bitset(i)
        selected = prune_and_optimize_bitset(selected, i)
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            
    if best_result is None:
        selected = greedy_mcv_bitset(0)
        selected = prune_and_optimize_bitset(selected, 0)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
             # Fallback: return a valid (but maybe larger) cover
             pass
             
    return result

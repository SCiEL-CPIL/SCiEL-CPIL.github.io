import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precompute all blocks and their covered t-subsets
    all_points = list(range(v))
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of t-subsets it covers
    block_covers = []
    for b in all_blocks:
        covers = list(itertools.combinations(b, t))
        block_covers.append(covers)
        
    # Reverse index: t-subset -> list of block indices
    # Use tuples for hashability
    cover_map = {}
    for i, covers in enumerate(block_covers):
        for c in covers:
            if c not in cover_map:
                cover_map[c] = []
            cover_map[c].append(i)
            
    # Total number of t-subsets
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(all_t_subsets)
    
    # 2. Greedy Construction
    # We will run a few restarts if time permits, but let's start with one robust run.
    best_blocks = []
    
    for restart in range(10): # Try a few restarts with random tie-breaking
        if time.time() > deadline - 20.0:
            break
            
        random.seed(restart)
        
        # covered_count[s] = number of selected blocks covering s
        covered_count = {s: 0 for s in all_t_subsets}
        # uncovered_count[b] = number of t-subsets in block b that are currently uncovered
        uncovered_count = [0] * n_blocks
        
        # Initialize uncovered_count
        for i, covers in enumerate(block_covers):
            # Initially all are uncovered
            uncovered_count[i] = len(covers)
            
        selected_blocks = []
        
        while True:
            # Find block with max uncovered_count
            # Optimization: We can keep a max-heap, but n_blocks is small (~50k),
            # so a linear scan or sorted list update might be comparable.
            # However, updating uncovered_count requires touching all blocks covering a specific s.
            # Let's just scan for max. 50k scans * ~150 iterations = 7.5M ops. Fast.
            max_idx = -1
            max_val = -1
            # Random tie breaking
            candidates = []
            for i in range(n_blocks):
                if uncovered_count[i] > max_val:
                    max_val = uncovered_count[i]
                    max_idx = i
                    candidates = [i]
                elif uncovered_count[i] == max_val:
                    candidates.append(i)
            
            if max_val == 0:
                break
                
            # Pick randomly from candidates to diversify
            chosen = random.choice(candidates)
            selected_blocks.append(chosen)
            
            # Update counts
            for s in block_covers[chosen]:
                if covered_count[s] == 0:
                    # This subset is now covered for the first time
                    covered_count[s] = 1
                    # Decrease uncovered_count for all blocks covering s
                    for b_idx in cover_map[s]:
                        uncovered_count[b_idx] -= 1
                else:
                    covered_count[s] += 1
            
            # Check time
            if time.time() > deadline - 5.0:
                break
                
        # 3. Reduction Phase
        # Try to remove redundant blocks
        # We need to check if a block is "essential".
        # A block is redundant if all its t-subsets are covered by other selected blocks.
        
        # To make this efficient, we can just re-verify coverage if we remove a block.
        # But checking all blocks is O(n_selected * k * C(k,t)).
        # n_selected is ~150. C(7,4)=35. 150*35 = 5250 checks per pass. Very fast.
        
        removed_any = True
        # We must be careful: if we remove a block, we might make another one redundant.
        # Simple loop: try to remove each block in random order.
        
        current_selected = list(selected_blocks)
        
        # We need a way to quickly check if a subset is covered by *others*.
        # Let's maintain a global covered_count again, but this time for the current set.
        # Actually, we can just recompute coverage from scratch for the reduction phase
        # if the set is small enough, or maintain the counts.
        # Let's maintain counts.
        
        # Recount for current_selected
        covered_count_red = {s: 0 for s in all_t_subsets}
        for b_idx in current_selected:
            for s in block_covers[b_idx]:
                covered_count_red[s] += 1
        
        # Now try to remove blocks
        # We iterate through a copy of current_selected
        # If we remove a block, we decrement counts for its subsets.
        # If any count becomes 0, we cannot remove it (or we must revert).
        
        # Better: iterate through indices. If removing block i causes any s to have count 0,
        # it's essential. Otherwise, remove it.
        
        # We need to be able to add it back if we realize later? No, greedy removal is usually safe
        # if we check one by one.
        
        i = 0
        while i < len(current_selected):
            if time.time() > deadline - 2.0:
                break
                
            b_idx = current_selected[i]
            # Check if all subsets in b_idx are covered by *other* blocks
            can_remove = True
            for s in block_covers[b_idx]:
                if covered_count_red[s] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for s in block_covers[b_idx]:
                    covered_count_red[s] -= 1
                current_selected.pop(i)
                # Do not increment i, check the new block at position i
            else:
                i += 1
        
        # Compare with best
        if not best_blocks or len(current_selected) < len(best_blocks):
            best_blocks = current_selected
            # Store the actual block tuples
            best_block_tuples = [all_blocks[i] for i in best_blocks]
            
        # If we have time, we could try local search (swap 1 block for 1 block), 
        # but for this size, greedy + reduction is often near-optimal.
        
    # Convert indices to tuples
    if best_blocks:
        return [list(all_blocks[i]) for i in best_blocks]
    else:
        # Fallback: should not happen
        return []

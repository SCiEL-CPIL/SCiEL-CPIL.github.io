import itertools
import random
import time

def construct(v, k, t):
    # Precompute blocks and their triple sets
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    num_triples = len(all_triples)
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)

    # 1. Initial Greedy Solution
    # We use a simple greedy to get a valid start
    def greedy_init():
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        selected = []
        
        while uncovered:
            # Find max gain
            max_gain = -1
            best_block = -1
            # Optimization: Only consider blocks with gain > 0
            # In a tight loop, this is fine for N=346k if done a few times
            # But for init, we can just scan.
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_block = i
            
            if max_gain == 0:
                break # Should not happen for valid covering problem
            
            selected.append(best_block)
            
            # Update gains
            for t_idx in block_triples[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        
        return selected

    # 2. Local Search (Minimization)
    # State: selected (list of block indices), cov (array of triple coverage counts)
    
    def is_redundant(block_idx, cov):
        """Check if block_idx can be removed without uncovering any triple."""
        for t_idx in block_triples[block_idx]:
            if cov[t_idx] == 1:
                return False
        return True

    def local_search(initial_selected, time_limit):
        # Initialize state
        selected = set(initial_selected)
        cov = [0] * num_triples
        for b in selected:
            for t_idx in block_triples[b]:
                cov[t_idx] += 1
        
        best_size = len(selected)
        best_set = set(selected)
        
        start_time = time.time()
        rng = random.Random(42)
        
        # We iterate until time limit
        while time.time() - start_time < time_limit:
            # Phase 1: Try to remove redundant blocks
            # We shuffle the order of selected blocks to avoid bias
            current_list = list(selected)
            rng.shuffle(current_list)
            
            removed_any = False
            for b in current_list:
                if b not in selected:
                    continue
                if is_redundant(b, cov):
                    # Remove it
                    selected.remove(b)
                    for t_idx in block_triples[b]:
                        cov[t_idx] -= 1
                    removed_any = True
            
            if removed_any:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = set(selected)
                # Continue trying to remove more, don't kick yet
                continue
            
            # Phase 2: No block is redundant. Local minimum.
            # Escape by adding a random block
            # Choose a block not in selected
            # To be efficient, we can pick a random block from all_blocks
            # and check if it's in selected. If so, pick another.
            # Or maintain a list of unselected? That's expensive to update.
            # Just pick random index.
            while True:
                new_b = rng.randint(0, num_blocks - 1)
                if new_b not in selected:
                    break
            
            # Add it
            selected.add(new_b)
            for t_idx in block_triples[new_b]:
                cov[t_idx] += 1
            
            # Immediately try to remove again (loop goes back to Phase 1)
            
        return best_set

    # Run Greedy Init
    initial_sol = greedy_init()
    
    # Run Local Search with remaining time
    # Leave 5s for final conversion and safety
    start_time = time.time()
    max_search_time = 110 - (time.time() - start_time)
    
    best_set = local_search(initial_sol, max_search_time)
    
    # Convert to list of blocks
    result = [list(all_blocks[i]) for i in best_set]
    return result

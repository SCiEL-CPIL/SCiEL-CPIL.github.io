import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Pre-computation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the list of triple indices it covers
    # This is faster to iterate over than bitmasks in Python for counts
    block_triples = [[] for _ in range(num_blocks)]
    for bi, b in enumerate(all_blocks):
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            block_triples[bi].append(idx)
            
    # Map each triple to the list of blocks that cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        for idx in block_triples[bi]:
            triple_to_blocks[idx].append(bi)

    # 2. Initialization
    # We use a greedy initialization to get a good starting point, 
    # then we run SA to optimize.
    
    def greedy_init(seed):
        random.seed(seed)
        uncovered = [1] * num_triples
        selected = []
        
        # We can pre-compute the "gain" for each block, but that changes as we select.
        # Simple greedy: pick block covering most uncovered.
        # To speed up, we can just pick a random uncovered triple and pick the best block for it.
        # This is the standard "greedy set cover" heuristic.
        
        while any(uncovered):
            # Find a random uncovered triple
            # To avoid O(N) scan every time, we could maintain a list of uncovered, 
            # but N is small (2024).
            uncovered_indices = [i for i, u in enumerate(uncovered) if u > 0]
            if not uncovered_indices:
                break
            target = random.choice(uncovered_indices)
            
            # Find best block covering target
            best_block = -1
            best_gain = -1
            candidates = triple_to_blocks[target]
            
            # Randomize order to break ties and add diversity
            random.shuffle(candidates)
            
            for bi in candidates:
                gain = 0
                for tri_idx in block_triples[bi]:
                    if uncovered[tri_idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block != -1:
                selected.append(best_block)
                for tri_idx in block_triples[best_block]:
                    uncovered[tri_idx] += 1 # Actually we want to mark as covered. 
                    # Let's use a boolean array or just count. 
                    # If count > 0, it's covered. 
                    # Wait, the logic above is slightly off. 
                    # Let's use a set of uncovered indices for efficiency? 
                    # No, list is fine.
        
        return selected

    # Let's refine greedy_init to be faster and correct
    def greedy_init_fast(seed):
        random.seed(seed)
        uncovered_mask = (1 << num_triples) - 1
        selected = []
        
        # Pre-compute block masks for greedy? 
        # We already have block_triples. 
        # Let's stick to the count array approach for consistency, 
        # but greedy is just a starting point.
        
        # Actually, let's just run a few greedy restarts and pick the best.
        # Then run SA on the best one.
        
        # To make greedy fast:
        # 1. Pick random uncovered triple
        # 2. Pick block covering it with max new coverage
        
        # We need a fast way to pick a random uncovered triple.
        # Since num_triples is ~2000, scanning is okay-ish, but let's optimize.
        # We can maintain a list of uncovered indices.
        
        uncovered_indices = list(range(num_triples))
        counts = [0] * num_triples # 0 means uncovered, >0 covered
        
        while uncovered_indices:
            target_idx = random.choice(uncovered_indices)
            candidates = triple_to_blocks[target_idx]
            random.shuffle(candidates)
            
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for tri_idx in block_triples[bi]:
                    if counts[tri_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block != -1:
                selected.append(best_block)
                # Update counts and uncovered list
                # Removing from list is O(N). 
                # We can just mark and filter later? 
                # Or just use a boolean array and scan. 
                # Scanning 2000 items is fast.
                
                for tri_idx in block_triples[best_block]:
                    if counts[tri_idx] == 0:
                        counts[tri_idx] = 1
                        # Remove from uncovered_indices
                        # This is O(N) per removal. 
                        # Total O(N^2) which might be slow for 2000.
                        # Let's just use a boolean array and find random uncovered by rejection sampling.
                        
            # Let's restart greedy with boolean array for speed
            break 
        
        # Re-implementing greedy with boolean array
        covered = [False] * num_triples
        selected = []
        
        while True:
            # Find random uncovered
            found = False
            while not found:
                idx = random.randint(0, num_triples - 1)
                if not covered[idx]:
                    found = True
                    target_idx = idx
            
            candidates = triple_to_blocks[target_idx]
            random.shuffle(candidates)
            
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for tri_idx in block_triples[bi]:
                    if not covered[tri_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block != -1:
                selected.append(best_block)
                for tri_idx in block_triples[best_block]:
                    covered[tri_idx] = True
            
            if all(covered):
                break
                
        return selected

    # Run a few greedy restarts
    best_greedy = []
    best_greedy_len = float('inf')
    
    for i in range(10):
        if time.time() > deadline - 60:
            break
        blocks = greedy_init_fast(42 + i)
        # Clean up redundancies
        # Simple redundancy removal
        counts = [0] * num_triples
        for bi in blocks:
            for tri_idx in block_triples[bi]:
                counts[tri_idx] += 1
        
        final_blocks = []
        for bi in blocks:
            redundant = True
            for tri_idx in block_triples[bi]:
                if counts[tri_idx] == 1:
                    redundant = False
                    break
            if not redundant:
                final_blocks.append(bi)
                for tri_idx in block_triples[bi]:
                    counts[tri_idx] -= 1
            
        if len(final_blocks) < best_greedy_len:
            best_greedy_len = len(final_blocks)
            best_greedy = final_blocks[:]
            
    # 3. Simulated Annealing
    # State: list of block indices
    # Move: Replace one block with another
    # Objective: Minimize number of blocks (by allowing removals if possible) 
    #            But SA usually works on fixed size. 
    #            Let's work on fixed size N, and try to minimize "uncovered" count first, 
    #            then minimize N? 
    #            Better: Start with best_greedy. 
    #            Allow moves: 
    #            1. Swap block i with block j. 
    #            2. Remove block i if it doesn't uncover anything. 
    #            3. Add block i if it reduces something? No, we want to minimize size.
            
    # Let's use a "repair" strategy or just accept moves that keep coverage valid.
    # If we keep coverage valid, we can only swap or remove.
    # If we allow temporary invalidity, we need a cost function.
    
    # Cost Function: 
    # If all covered: Cost = len(blocks)
    # If not all covered: Cost = len(blocks) + 100 * (num_uncovered)
    
    current_state = best_greedy[:]
    counts = [0] * num_triples
    for bi in current_state:
        for tri_idx in block_triples[bi]:
            counts[tri_idx] += 1
            
    def get_cost(state, counts):
        uncovered = 0
        for c in counts:
            if c == 0:
                uncovered += 1
        if uncovered > 0:
            return len(state) + 100 * uncovered
        return len(state)
        
    current_cost = get_cost(current_state, counts)
    best_state = current_state[:]
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.9995
    T_min = 0.1
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        if T < T_min:
            break
            
        # Choose a move
        if len(current_state) == 0:
            break
            
        move_type = random.random()
        
        if move_type < 0.5 and len(current_state) > 1:
            # Swap
            idx = random.randint(0, len(current_state) - 1)
            old_block = current_state[idx]
            
            # Pick new block
            # To make it interesting, pick a block that covers some of the same triples?
            # Or just random.
            new_block = random.randint(0, num_blocks - 1)
            
            if new_block == old_block:
                continue
                
            # Calculate delta cost
            # Remove old
            old_triples = block_triples[old_block]
            for tri_idx in old_triples:
                counts[tri_idx] -= 1
                
            new_triples = block_triples[new_block]
            for tri_idx in new_triples:
                counts[tri_idx] += 1
                
            new_cost = get_cost(current_state, counts)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_state[idx] = new_block
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                for tri_idx in old_triples:
                    counts[tri_idx] += 1
                for tri_idx in new_triples:
                    counts[tri_idx] -= 1
                no_improve += 1
                
        else:
            # Remove
            idx = random.randint(0, len(current_state) - 1)
            old_block = current_state[idx]
            old_triples = block_triples[old_block]
            
            # Check if removal is valid (all counts > 1)
            valid = True
            for tri_idx in old_triples:
                if counts[tri_idx] == 1:
                    valid = False
                    break
                    
            if valid:
                for tri_idx in old_triples:
                    counts[tri_idx] -= 1
                current_state.pop(idx)
                
                new_cost = get_cost(current_state, counts)
                # Cost should decrease by 1
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        T *= cooling_rate
        
    # Final Cleanup: Remove any remaining redundant blocks
    final_state = best_state[:]
    counts = [0] * num_triples
    for bi in final_state:
        for tri_idx in block_triples[bi]:
            counts[tri_idx] += 1
            
    final_blocks = []
    for bi in final_state:
        redundant = True
        for tri_idx in block_triples[bi]:
            if counts[tri_idx] == 1:
                redundant = False
                break
        if not redundant:
            final_blocks.append(bi)
            for tri_idx in block_triples[bi]:
                counts[tri_idx] -= 1
                
    # Convert to list of lists
    return [list(all_blocks[bi]) for bi in final_blocks]

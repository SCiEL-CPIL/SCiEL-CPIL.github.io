import itertools
import random
import time
import sys

def construct(v, k, t):
    # Parameters for C(24, 7, 3)
    # We assume v, k, t match the task description for performance, 
    # but the logic is generic enough for small v.
    
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    
    num_blocks = len(all_blocks)
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute block_triples: for each block, list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Precompute triple_to_blocks: for each triple, list of block indices covering it
    # This is needed for efficient delta calculation when flipping a block.
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)
            
    # Local Search (WalkSAT variant)
    # State: boolean array `selected` of size num_blocks
    # Score: number of uncovered triples.
    # Move: Flip one block.
    # Acceptance: 
    #   - If score decreases, accept.
    #   - If score increases, accept with probability p (or always accept in random walk phases).
    #   - We want to minimize score to 0 and then minimize number of selected blocks.
    
    # Optimization: 
    # We track `cover_count[t]` = number of selected blocks covering triple t.
    # A triple is covered if cover_count[t] > 0.
    # Uncovered count = sum(1 for c in cover_count if c == 0).
    
    # To speed up, we can track the number of uncovered triples directly.
    
    best_solution = None
    best_uncovered = float('inf')
    best_num_blocks = float('inf')
    
    start_time = time.time()
    max_time = 115
    
    # Random Walk Parameters
    # We will run multiple restarts.
    # In each restart, we start with a random set of blocks or empty.
    # Starting with empty is safer for minimizing count, but might take longer to cover.
    # Starting with a greedy solution might be a good starting point.
    # Let's try starting with a greedy solution and then doing local search to reduce count.
    
    def greedy_init():
        """Greedy construction to get a valid starting point."""
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        selected = [False] * num_blocks
        blocks_used = []
        
        # Simple greedy: pick block with max gain
        # To be faster, we don't do full random restarts here, just one greedy pass.
        while uncovered:
            max_gain = -1
            best_idx = -1
            # Optimization: only look at blocks that might have high gain?
            # For simplicity and speed in init, scan all.
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
            
            if max_gain == 0:
                break # Should not happen for valid covering numbers, but safety
                
            selected[best_idx] = True
            blocks_used.append(best_idx)
            
            # Update gains
            for t_idx in block_triples[best_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        if gains[b_prime] < 0: # Safety
                             gains[b_prime] = 0
                             
        return selected, blocks_used

    def evaluate(selected):
        """Calculate uncovered triples and count of selected blocks."""
        cover_count = [0] * num_triples
        num_sel = 0
        for i in range(num_blocks):
            if selected[i]:
                num_sel += 1
                for t_idx in block_triples[i]:
                    cover_count[t_idx] += 1
        uncovered = sum(1 for c in cover_count if c == 0)
        return uncovered, num_sel, cover_count

    def local_search(initial_selected, time_limit):
        """
        Perform local search starting from initial_selected.
        Returns best (uncovered, num_blocks, selected) found.
        """
        selected = initial_selected[:]
        uncovered, num_sel, cover_count = evaluate(selected)
        
        best_uncovered = uncovered
        best_num_sel = num_sel
        best_selected = selected[:]
        
        # If we start with a valid solution (uncovered == 0), we try to reduce num_sel.
        # If not, we try to reduce uncovered first.
        
        # Strategy: 
        # Phase 1: Minimize uncovered.
        # Phase 2: Minimize num_blocks (only if uncovered == 0).
        
        # We can combine them into a single objective: 
        # Score = uncovered * large_constant + num_blocks
        # Or just prioritize uncovered.
        
        # Let's use a simple random walk with restarts inside the local search.
        
        # To be efficient, we pre-calculate the delta for flipping a block.
        # Delta in uncovered count when flipping block i:
        # If selected[i] is True (removing it):
        #   For each t in block_triples[i]:
        #     if cover_count[t] == 1: uncovered increases by 1
        #     else: no change
        # If selected[i] is False (adding it):
        #   For each t in block_triples[i]:
        #     if cover_count[t] == 0: uncovered decreases by 1
        #     else: no change
        
        # We also care about num_blocks.
        # If we remove a block, num_blocks decreases by 1.
        # If we add a block, num_blocks increases by 1.
        
        # Objective: Minimize (uncovered, num_blocks) lexicographically.
        
        # We will use a "simulated annealing" like approach or just random hill climbing with random restarts.
        # Given time constraints, random hill climbing with random restarts is robust.
        
        current_uncovered = uncovered
        current_num_sel = num_sel
        current_cover = cover_count[:]
        
        # We will run for a short duration, then restart.
        # Or just run until time limit for this specific restart.
        # Let's do multiple short restarts to explore more.
        
        # Actually, let's just run one long local search but with random "kicks" (random flips) when stuck.
        
        stuck_count = 0
        max_stuck = 5000 # If no improvement for 5000 moves, do a random kick
        
        while time.time() < time_limit:
            if current_uncovered == 0 and current_num_sel < best_num_sel:
                best_num_sel = current_num_sel
                best_uncovered = 0
                best_selected = selected[:]
                
            # Pick a random block to flip
            i = random.randint(0, num_blocks - 1)
            
            # Calculate delta
            delta_uncovered = 0
            if selected[i]:
                # Removing block i
                # Check if any triple becomes uncovered
                for t_idx in block_triples[i]:
                    if current_cover[t_idx] == 1:
                        delta_uncovered += 1
                # num_blocks decreases by 1
                delta_num_sel = -1
            else:
                # Adding block i
                # Check if any triple becomes covered
                for t_idx in block_triples[i]:
                    if current_cover[t_idx] == 0:
                        delta_uncovered -= 1
                # num_blocks increases by 1
                delta_num_sel = 1
                
            new_uncovered = current_uncovered + delta_uncovered
            new_num_sel = current_num_sel + delta_num_sel
            
            # Acceptance Criteria:
            # 1. If new_uncovered < current_uncovered: Accept (Improvement in coverage)
            # 2. If new_uncovered == current_uncovered and new_num_sel < current_num_sel: Accept (Improvement in size)
            # 3. If new_uncovered == 0 and current_uncovered == 0:
            #      Accept if new_num_sel < current_num_sel
            #      Or accept with probability p if new_num_sel > current_num_sel (to escape local min in size)
            # 4. If new_uncovered > current_uncovered:
            #      Accept with small probability (to escape coverage local minima)
            
            accept = False
            if new_uncovered < current_uncovered:
                accept = True
            elif new_uncovered == current_uncovered:
                if new_num_sel < current_num_sel:
                    accept = True
                else:
                    # Same coverage, larger size. Only accept if we are stuck or to explore.
                    # If uncovered is 0, we want to minimize size.
                    # If uncovered > 0, we don't care about size as much yet.
                    if current_uncovered == 0:
                         # We have a valid solution, try to shrink.
                         # Randomly accept worse size to escape local min?
                         # Or just reject to keep shrinking.
                         # Let's reject to keep shrinking unless we are stuck.
                         if stuck_count > 100:
                             accept = True
                    else:
                         # Not valid yet. Accepting a larger set might help cover more later?
                         # Usually adding blocks helps coverage.
                         # But we just calculated delta_uncovered == 0.
                         # So adding a block that doesn't cover new triples is useless for coverage.
                         # It only increases size. Reject.
                         accept = False
            else:
                # new_uncovered > current_uncovered
                # Deterioration in coverage.
                # Accept with small probability to escape local minima.
                if random.random() < 0.05:
                    accept = True
                    
            if accept:
                # Apply move
                selected[i] = not selected[i]
                for t_idx in block_triples[i]:
                    if selected[i]:
                        current_cover[t_idx] += 1
                    else:
                        current_cover[t_idx] -= 1
                current_uncovered = new_uncovered
                current_num_sel = new_num_sel
                stuck_count = 0
            else:
                stuck_count += 1
                
            # Random Kick if stuck
            if stuck_count > max_stuck:
                # Perform a random kick: flip 3-5 random blocks
                for _ in range(random.randint(3, 5)):
                    j = random.randint(0, num_blocks - 1)
                    selected[j] = not selected[j]
                    for t_idx in block_triples[j]:
                        if selected[j]:
                            current_cover[t_idx] += 1
                        else:
                            current_cover[t_idx] -= 1
                # Recalculate stats
                current_uncovered, current_num_sel, current_cover = evaluate(selected)
                stuck_count = 0
                
        return best_uncovered, best_num_sel, best_selected

    # Main Loop: Multiple Restarts
    # 1. Greedy init
    # 2. Local search
    # 3. Repeat with different random seeds/initializations
    
    # We'll run a few greedy inits with different randomizations?
    # The greedy above is deterministic. Let's add some randomness to greedy init.
    
    def greedy_init_random(seed):
        random.seed(seed)
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        selected = [False] * num_blocks
        blocks_used = []
        
        while uncovered:
            # Find top candidates
            max_gain = -1
            candidates = []
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    candidates = [i]
                elif gains[i] == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            b = random.choice(candidates)
            selected[b] = True
            blocks_used.append(b)
            
            for t_idx in block_triples[b]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        if gains[b_prime] < 0:
                            gains[b_prime] = 0
                            
        return selected, blocks_used

    # Run local search from several random greedy starts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > max_time:
            break
            
        # Time budget per restart
        # If 10 restarts, ~11s each.
        restart_limit = start_time + (max_time - start_time) * (i + 1) / num_restarts
        
        # Random seed for this restart
        seed = i * 12345 + 67890
        random.seed(seed)
        
        # Get initial solution
        selected_init, _ = greedy_init_random(seed)
        
        # Run local search
        u, n, sol = local_search(selected_init, restart_limit)
        
        if u < best_uncovered or (u == best_uncovered and n < best_num_blocks):
            best_uncovered = u
            best_num_blocks = n
            best_solution = sol
            
    # Final check: If best_solution is not valid (u > 0), we might need to finish it with greedy?
    # But local search should find a valid one if one exists.
    # If best_solution is None, fallback to simple greedy.
    if best_solution is None or best_uncovered > 0:
        # Fallback: Run one long greedy
        selected_fb, _ = greedy_init_random(0)
        # Check if valid
        u_fb, n_fb, _ = evaluate(selected_fb)
        if u_fb == 0:
            best_solution = selected_fb
            best_num_blocks = n_fb
        else:
            # Emergency: just return the greedy blocks, even if invalid? 
            # No, must be valid.
            # If greedy failed (unlikely), we have a problem.
            # For now, assume greedy works.
            best_solution = selected_fb
            
    # Convert boolean array to list of blocks
    result = []
    for i in range(num_blocks):
        if best_solution[i]:
            result.append(list(all_blocks[i]))
            
    return result

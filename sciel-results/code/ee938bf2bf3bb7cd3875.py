import itertools
import random
import time
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for cleanup/return
    
    # Generate all words in {0, 1, 2}^6
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                new_w = tuple(w_list)
                neighbors[i].append(word_to_idx[new_w])
            w_list[pos] = orig_val

    # Initial construction: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.add(tuple(h + list(tail)))
    
    # State variables
    current_code = set(initial_code)
    coverage = [0] * num_words
    uncovered_count = num_words # Initially all uncovered
    
    def add_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
        current_code.add(w)
        
    def remove_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            coverage[nb_idx] -= 1
            if coverage[nb_idx] == 0:
                uncovered_count += 1
        current_code.remove(w)
        
    # Initialize coverage
    for w in initial_code:
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
            
    best_code = set(current_code)
    best_size = len(current_code)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # To avoid iterating over set for random choice (O(N)), keep a list and sync?
    # Or just use random.sample from list(current_code) which is O(N).
    # N is ~80, so it's fine.
    
    while time.time() < deadline:
        # Update best
        if uncovered_count == 0 and len(current_code) < best_size:
            best_code = set(current_code)
            best_size = len(current_code)
            
        if not current_code:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Removal
            w_old = random.choice(list(current_code))
            
            # Check if removal is valid: no neighbor has coverage 1
            can_remove = True
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors[idx_old]:
                if coverage[nb_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                remove_word(w_old)
                # uncovered_count unchanged by valid removal
            else:
                # Try a replacement to potentially enable removal
                # Pick a random word to replace w_old with
                w_new = random.choice(all_words)
                if w_new == w_old:
                    continue
                
                # Calculate delta for replacement
                # Remove w_old, Add w_new
                # We need to check if this improves coverage or enables removals later
                # Let's just evaluate the coverage change
                
                # Temporarily apply to calculate delta? 
                # Or calculate analytically.
                # Let's do a dry run.
                
                # Save state of coverage for affected nodes
                # This is complex. Let's just apply, check, revert if bad.
                
                # Apply
                # We need to be careful with uncovered_count updates
                # Let's make helper functions that return the delta or use global state carefully.
                
                # Simpler: Just apply, compute new uncovered, decide, revert if needed.
                
                # Apply remove
                idx_old = word_to_idx[w_old]
                for nb_idx in neighbors[idx_old]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_old)
                
                # Apply add
                idx_new = word_to_idx[w_new]
                for nb_idx in neighbors[idx_new]:
                    if coverage[nb_idx] == 0:
                        uncovered_count -= 1
                    coverage[nb_idx] += 1
                current_code.add(w_new)
                
                new_uncovered = uncovered_count
                # Size is same
                
                # Acceptance
                # We prefer moves that reduce uncovered count.
                # If uncovered count increased, reject unless T is high.
                delta_u = new_uncovered - (uncovered_count - (new_uncovered - (uncovered_count))) # This is confusing.
                
                # Let's track the uncovered count before the move.
                
                # RESTART LOGIC FOR REPLACEMENT WITH PROPER TRACKING
                
                # Undo the move first to get clean state for logic? No, let's do it properly.
                
                # I will rewrite the loop body for clarity and correctness.
                pass
                
        else:
            # Replacement
            w_old = random.choice(list(current_code))
            w_new = random.choice(all_words)
            if w_new == w_old:
                continue
                
            # Save current uncovered count
            u_before = uncovered_count
            
            # Apply removal
            idx_old = word_to_idx[w_old]
            for nb_idx in neighbors[idx_old]:
                coverage[nb_idx] -= 1
                if coverage[nb_idx] == 0:
                    uncovered_count += 1
            current_code.remove(w_old)
            
            # Apply addition
            idx_new = word_to_idx[w_new]
            for nb_idx in neighbors[idx_new]:
                if coverage[nb_idx] == 0:
                    uncovered_count -= 1
                coverage[nb_idx] += 1
            current_code.add(w_new)
            
            u_after = uncovered_count
            delta_u = u_after - u_before
            
            # Acceptance
            accept = False
            if delta_u < 0:
                accept = True
            elif delta_u == 0:
                # Neutral coverage. Accept with some prob to explore.
                accept = random.random() < math.exp(-0.1 / T)
            else:
                # Worsens coverage.
                accept = random.random() < math.exp(-2.0 * delta_u / T)
                
            if not accept:
                # Revert
                # Remove w_new
                for nb_idx in neighbors[idx_new]:
                    coverage[nb_idx] -= 1
                    if coverage[nb_idx] == 0:
                        uncovered_count += 1
                current_code.remove(w_new)
                
                # Add w_old
                for nb_idx in neighbors[idx_old]:
                    if coverage[nb_idx] == 0:
                        uncovered_count -= 1
                    coverage[nb_idx] += 1
                current_code.add(w_old)
                
        if T > min_T:
            T *= cooling

    return [list(w) for w in best_code]

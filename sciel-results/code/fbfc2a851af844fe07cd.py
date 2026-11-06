import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    N = 1000
    best_set = []
    best_size = 0
    
    def evaluate_candidate(x, chosen, counts):
        """
        Combined check: returns (is_valid, tight_count) for candidate x.
        Single pass through chosen to compute both validity and tightness.
        """
        tight = 0
        # Check 2*x
        c2x = counts.get(2 * x, 0)
        if c2x >= 2:
            return (False, 0)
        if c2x == 1:
            tight += 1
        
        # Check x + y for each y in chosen
        for y in chosen:
            s = x + y
            c = counts.get(s, 0)
            if c >= 2:
                return (False, 0)
            if c == 1:
                tight += 1
        
        return (True, tight)

    def add_to_set(x, chosen, counts):
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        chosen.append(x)

    def remove_from_set(x, chosen, counts):
        counts[2 * x] -= 1
        if counts[2 * x] == 0:
            del counts[2 * x]
        for y in chosen:
            if y == x:
                continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        chosen.remove(x)

    def run_greedy(noise_scale=0.0):
        chosen = []
        counts = defaultdict(int)
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                break
                
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                valid, tight = evaluate_candidate(x, chosen, counts)
                if valid:
                    score = tight + (random.random() * noise_scale if noise_scale > 0 else 0)
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is None:
                break
                
            add_to_set(best_x, chosen, counts)
            candidates.remove(best_x)
            
        return chosen

    def local_search(chosen, counts, max_iters=200):
        """
        Try to improve by removing one element and adding another.
        Returns improved set or original if no improvement found.
        """
        if time.time() > deadline:
            return chosen
        
        chosen_copy = list(chosen)
        counts_copy = dict(counts)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing a random element and adding a better one
            if len(chosen_copy) < 2:
                break
                
            idx = random.randint(0, len(chosen_copy) - 1)
            removed = chosen_copy[idx]
            
            # Temporarily remove
            remove_from_set(removed, chosen_copy, counts_copy)
            
            # Find best replacement
            best_add = None
            best_score = float('inf')
            
            for x in range(1, N + 1):
                if x in chosen_copy:
                    continue
                valid, tight = evaluate_candidate(x, chosen_copy, counts_copy)
                if valid:
                    score = tight + random.random() * 0.3
                    if score < best_score:
                        best_score = score
                        best_add = x
            
            if best_add is not None:
                add_to_set(best_add, chosen_copy, counts_copy)
                improved = True
            else:
                # Put back what we removed
                add_to_set(removed, chosen_copy, counts_copy)
            
            if not improved:
                break
        
        return chosen_copy

    # Phase 1: Deterministic Min-Cost Greedy
    initial_set = run_greedy(noise_scale=0.0)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Build counts for best_set
    best_counts = defaultdict(int)
    for x in best_set:
        best_counts[2 * x] = best_counts.get(2 * x, 0) + 1
        for y in best_set:
            if x < y:
                s = x + y
                best_counts[s] = best_counts.get(s, 0) + 1
    
    # Phase 2: Randomized greedy restarts with local search
    while time.time() < deadline - 5:  # Reserve 5s for final local search
        current_set = run_greedy(noise_scale=0.5)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # Rebuild counts
            best_counts = defaultdict(int)
            for x in best_set:
                best_counts[2 * x] = best_counts.get(2 * x, 0) + 1
                for y in best_set:
                    if x < y:
                        s = x + y
                        best_counts[s] = best_counts.get(s, 0) + 1
        
        # Try local search on current best
        if time.time() < deadline - 2:
            improved = local_search(list(best_set), dict(best_counts), max_iters=50)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
                best_counts = defaultdict(int)
                for x in best_set:
                    best_counts[2 * x] = best_counts.get(2 * x, 0) + 1
                    for y in best_set:
                        if x < y:
                            s = x + y
                            best_counts[s] = best_counts.get(s, 0) + 1
    
    # Final local search pass
    if time.time() < deadline - 1:
        improved = local_search(list(best_set), dict(best_counts), max_iters=100)
        if len(improved) > best_size:
            best_set = improved
    
    return best_set

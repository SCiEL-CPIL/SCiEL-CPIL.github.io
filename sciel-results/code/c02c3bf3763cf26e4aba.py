import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build balls array: balls[w] = [w, neighbors...]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping: for each word w, which ball centers cover it
    # inv[w] = list of center indices c such that w is in balls[c]
    from collections import defaultdict
    inv = [defaultdict(int) for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w][c] += 1
            
    # Convert to lists for faster iteration
    inv_lists = [list(inv[w].keys()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    # Use multiple random restarts with greedy + pruning + local search
    np.random.seed(42)
    random.seed(42)
    
    def greedy_construct(seed_offset=0):
        """Randomized greedy construction with incremental updates."""
        rng = np.random.RandomState(42 + seed_offset)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.full(total, ball_size, dtype=np.int32)
        
        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find the best candidate among active centers
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            # Pick randomly among max-count active candidates for diversity
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            # Update cover counts and candidate counts
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    # Decrement cand_count for all centers that cover w
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
        
        return selected, cover_count, uncovered_count
    
    def prune(selected, cover_count):
        """Iteratively remove codewords that are not needed."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected
    
    def local_search(selected, cover_count):
        """Enhanced swap-based optimization."""
        improved = True
        while improved:
            improved = False
            # Recompute cover counts from scratch to be safe after swaps/removals
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
                
            selected_set = set(selected)
            
            # Identify critical codewords and their unique words
            critical_candidates = []
            for c in selected:
                unique_words = [w for w in balls[c] if cover_count[w] == 1]
                if unique_words:
                    critical_candidates.append((c, unique_words))
            
            # Sort critical codewords by number of unique words (hardest to replace first)
            critical_candidates.sort(key=lambda x: len(x[1]), reverse=True)
            
            # Try to find a swap for any critical codeword
            for c, unique_words in critical_candidates:
                if not unique_words:
                    continue
                    
                # Find intersection of inv_lists for all unique words
                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])
                    
                # Filter out already selected candidates
                available_candidates = [c_prime for c_prime in common_candidates if c_prime not in selected_set]
                
                if not available_candidates:
                    continue
                
                # Score candidates
                def candidate_score(c_prime):
                    score = 0
                    for w in balls[c_prime]:
                        if cover_count[w] == 2:
                            score -= 1
                        elif cover_count[w] >= 3:
                            score += 1
                    return score
                
                available_candidates.sort(key=candidate_score, reverse=True)
                c_prime = available_candidates[0]
                
                # Verify the swap is valid
                valid_swap = True
                for w in unique_words:
                    if w not in balls[c_prime]:
                        valid_swap = False
                        break
                
                if valid_swap:
                    selected.remove(c)
                    selected.append(c_prime)
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    improved = True
                    break
            
            # After swaps, try to remove redundant codewords again
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
                
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                else:
                    new_selected.append(c)
                
            if len(new_selected) < len(selected):
                selected = new_selected
                improved = True
        
        return selected
    
    def simulated_annealing(selected, cover_count, time_budget):
        """Simulated annealing approach to escape local minima."""
        start_time = time.time()
        T = 1.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.95
        rng = np.random.RandomState(42)
        
        current_size = len(selected)
        best_selected = selected.copy()
        best_size = current_size
        
        # Compute uncovered count
        uncovered_count = np.sum(cover_count == 0)
        
        while time.time() - start_time < time_budget and T > T_min:
            # Generate a move: add, remove, or swap
            move_type = rng.choice(['add', 'remove', 'swap'], p=[0.3, 0.3, 0.4])
            
            new_selected = selected.copy()
            new_cover_count = cover_count.copy()
            
            if move_type == 'add':
                # Add a random codeword
                c = rng.randint(total)
                if c not in new_selected:
                    new_selected.append(c)
                    for w in balls[c]:
                        new_cover_count[w] += 1
            
            elif move_type == 'remove':
                # Remove a random codeword
                if len(new_selected) > 0:
                    idx = rng.randint(len(new_selected))
                    c = new_selected[idx]
                    for w in balls[c]:
                        new_cover_count[w] -= 1
                    new_selected.pop(idx)
            
            else:  # swap
                # Swap a random codeword with another
                if len(new_selected) > 0:
                    idx = rng.randint(len(new_selected))
                    c_old = new_selected[idx]
                    c_new = rng.randint(total)
                    if c_new not in new_selected:
                        for w in balls[c_old]:
                            new_cover_count[w] -= 1
                        for w in balls[c_new]:
                            new_cover_count[w] += 1
                        new_selected[idx] = c_new
            
            # Compute objective: size + penalty for uncovered words
            new_uncovered = np.sum(new_cover_count == 0)
            current_obj = current_size + 10 * uncovered_count
            new_obj = len(new_selected) + 10 * new_uncovered
            
            delta_obj = new_obj - current_obj
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / T):
                selected = new_selected
                cover_count = new_cover_count
                current_size = len(selected)
                uncovered_count = new_uncovered
                
                if len(selected) < best_size and uncovered_count == 0:
                    best_size = len(selected)
                    best_selected = selected.copy()
            
            # Cool down
            T *= cooling_rate
        
        return best_selected, best_size
    
    # Main search loop
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        seed_offset = restart_count * 1000
        
        # Greedy construction
        selected, cover_count, uncovered_count = greedy_construct(seed_offset)
        
        if uncovered_count > 0:
            continue
        
        # Pruning
        selected = prune(selected, cover_count)
        
        # Local search
        selected = local_search(selected, cover_count)
        
        # Recompute cover count after local search
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Simulated annealing if we have time
        remaining_time = deadline - 10.0 - time.time()
        if remaining_time > 5.0 and len(selected) > 486:
            selected, sa_size = simulated_annealing(selected, cover_count, min(remaining_time * 0.5, 15.0))
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we've found a good solution, do more restarts
        if best_size <= 490 and time.time() < deadline - 10.0:
            continue
        
        # Early stop if time is running low
        if time.time() > deadline - 8.0:
            break
    
    # Convert to list of digit lists
    if best_set is None:
        # Fallback: return a large valid set
        best_set = list(range(total))
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

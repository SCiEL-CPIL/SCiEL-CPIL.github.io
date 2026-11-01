import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
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
    
    def greedy_construct():
        """Randomized greedy set cover construction"""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
        
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
            best_idx = candidates[np.random.randint(len(candidates))]
            
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
            
            if time.time() > deadline - 10.0:
                break
        
        return selected, uncovered_count
    
    def prune_redundant(selected):
        """Remove codewords that are not needed"""
        if not selected:
            return selected
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3.0:
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
    
    def local_search_swap(selected):
        """Swap-based local search optimization"""
        if not selected:
            return selected
        
        improved = True
        while improved and time.time() < deadline - 2.0:
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
            
            # Try to find a swap for any critical codeword
            for c, unique_words in critical_candidates:
                # Find intersection of inv_lists for all unique words
                # These are the candidates c' that cover ALL unique words
                if not unique_words:
                    continue
                    
                # Start with inv_lists of the first word
                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])
                    
                # Check if any common candidate is NOT in selected
                for c_prime in common_candidates:
                    if c_prime not in selected_set:
                        # Perform swap
                        selected.remove(c)
                        selected.append(c_prime)
                        selected_set.remove(c)
                        selected_set.add(c_prime)
                        improved = True
                        break  # Restart the loop to re-evaluate criticals
                if improved:
                    break
            
            # After swaps, try to remove redundant codewords again
            if time.time() < deadline - 2.0:
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
    
    def dual_guided_set_cover():
        """
        Dual-guided set cover: uses shadow prices (inverse of remaining candidate count)
        to prioritize hard-to-cover words. Different optimization paradigm than pure greedy.
        """
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
        
        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find uncovered words and compute their "difficulty" (inverse of number of candidates that can cover them)
            uncovered_words = np.where(uncovered)[0]
            if len(uncovered_words) == 0:
                break
            
            # For each uncovered word, compute difficulty = 1 / (number of active candidates that cover it)
            # Words with fewer covering candidates are "harder" to cover
            difficulties = np.zeros(len(uncovered_words), dtype=np.float64)
            for i, w in enumerate(uncovered_words):
                num_active_covering = sum(1 for c in inv_lists[w] if active[c])
                difficulties[i] = 1.0 / max(num_active_covering, 1)
            
            # Pick the hardest uncovered word (highest difficulty)
            hardest_idx = np.argmax(difficulties)
            hardest_word = uncovered_words[hardest_idx]
            
            # Among active candidates that cover the hardest word, pick the one with highest cand_count
            candidates_covering = [c for c in inv_lists[hardest_word] if active[c]]
            if not candidates_covering:
                break
            
            # Pick candidate with highest coverage
            best_candidate = max(candidates_covering, key=lambda c: cand_count[c])
            
            selected.append(best_candidate)
            active[best_candidate] = False
            
            # Update cover counts and candidate counts
            for w in balls[best_candidate]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    uncovered[w] = False
                    # Decrement cand_count for all centers that cover w
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
            
            if time.time() > deadline - 10.0:
                break
        
        return selected, uncovered_count
    
    def add_remove_local_search(selected):
        """
        Add-remove local search: explore neighborhood by removing 2 and adding 1.
        This explores a different neighborhood than single-swap search.
        """
        if not selected:
            return selected
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Recompute cover counts
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
            
            selected_set = set(selected)
            n_selected = len(selected)
            
            # Try removing pairs of codewords and see if we can add one to maintain coverage
            # Only try a subset of pairs for efficiency
            max_pairs_to_try = min(n_selected * (n_selected - 1) // 2, 500)
            np.random.seed(int(time.time() * 1000) % 100000)
            
            pairs_tried = 0
            for i in range(n_selected):
                if pairs_tried >= max_pairs_to_try:
                    break
                for j in range(i + 1, n_selected):
                    if pairs_tried >= max_pairs_to_try:
                        break
                    
                    c1, c2 = selected[i], selected[j]
                    
                    # Check if removing both would leave any word uncovered
                    # Words that become uncovered are those with cover_count == 1 in balls[c1] or balls[c2]
                    potentially_uncovered = set()
                    for w in balls[c1]:
                        if cover_count[w] <= 1:
                            potentially_uncovered.add(w)
                    for w in balls[c2]:
                        if cover_count[w] <= 1:
                            potentially_uncovered.add(w)
                    
                    if not potentially_uncovered:
                        # Both can be removed without issue - just remove them
                        selected.remove(c1)
                        selected.remove(c2)
                        selected_set.discard(c1)
                        selected_set.discard(c2)
                        improved = True
                        break
                    
                    # Try to find a single codeword that covers all potentially_uncovered words
                    # and is not in selected
                    if len(potentially_uncovered) <= 4:  # Only try if manageable
                        common_candidates = set(inv_lists[list(potentially_uncovered)[0]])
                        for w in list(potentially_uncovered)[1:]:
                            common_candidates.intersection_update(inv_lists[w])
                        
                        for c_prime in common_candidates:
                            if c_prime not in selected_set:
                                # Perform add-remove: remove c1, c2, add c_prime
                                selected.remove(c1)
                                selected.remove(c2)
                                selected.append(c_prime)
                                selected_set.discard(c1)
                                selected_set.discard(c2)
                                selected_set.add(c_prime)
                                improved = True
                                break
                        if improved:
                            break
                    
                    pairs_tried += 1
                
                if improved:
                    break
            
            # After add-remove, try to remove redundant codewords
            if time.time() < deadline - 2.0:
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
    
    # Main loop: try multiple strategies with random restarts
    while time.time() < deadline - 5.0:
        # Alternate between greedy and dual-guided
        if random.random() < 0.5:
            selected, uncovered_count = greedy_construct()
        else:
            selected, uncovered_count = dual_guided_set_cover()
        
        # Check if valid
        if uncovered_count > 0:
            continue
        
        # Pruning: iteratively remove codewords that are not needed
        selected = prune_redundant(selected)
        
        # Local Search: Swap-based optimization
        selected = local_search_swap(selected)
        
        # Add-remove local search
        selected = add_remove_local_search(selected)
        
        # Final pruning
        selected = prune_redundant(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Early stop if we've found a very good solution and time is running low
        if best_size <= 486 and time.time() > deadline - 15.0:
            break
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

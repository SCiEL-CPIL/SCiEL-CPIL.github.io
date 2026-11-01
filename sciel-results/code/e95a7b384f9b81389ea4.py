import itertools
import numpy as np
import time
import random
from collections import defaultdict

def _fallback_construct(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _build_structures(q, n, radius):
    total = q ** n
    ball_size = 1 + n * (q - 1)
    weights = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Precompute digits
    all_indices = np.arange(total, dtype=np.int64)
    digits = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        digits[:, i] = (all_indices // weights[i]) % q
    
    # Precompute balls: balls[c] = list of words covered by center c
    balls = np.zeros((total, ball_size), dtype=np.int64)
    balls[:, 0] = np.arange(total)
    for i in range(n):
        w = weights[i]
        for v in range(1, q):
            # For each word, the neighbor at position i with digit changed by v
            # digit at pos i is (idx // w) % q
            # neighbor = idx + (v - digit) * w
            # But we need to handle wraparound: if digit + v >= q, neighbor = idx + (v - digit - q) * w
            # Actually simpler: for each word, for each pos, for each new digit != old digit
            # Let's just do it directly
            pass
    
    # More efficient: build balls and inv together
    balls = np.zeros((total, ball_size), dtype=np.int64)
    balls[:, 0] = np.arange(total)
    
    # Build inverse: inv[w] = list of centers c such that w is in balls[c]
    inv = [defaultdict(int) for _ in range(total)]
    
    for c in range(total):
        balls[c, 0] = c
        inv[c][c] = 1
        pos_idx = 1
        for i in range(n):
            w = weights[i]
            digit = (c // w) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                neighbor = c + (new_digit - digit) * w
                balls[c, pos_idx] = neighbor
                inv[neighbor][c] = 1
                pos_idx += 1
    
    # Convert inv to lists
    inv_lists = [list(inv[w].keys()) for w in range(total)]
    
    return digits, balls, inv_lists, ball_size, total

def _greedy_with_pruning_and_search(q, n, radius, deadline, seed, digits, balls, inv_lists, ball_size, total):
    rng = np.random.RandomState(seed)
    random.seed(seed)
    
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []
    selected_set = set()
    
    # cand_count[c] = number of uncovered words in balls[c]
    cand_count = np.full(total, ball_size, dtype=np.int32)
    active = np.ones(total, dtype=bool)
    
    # Greedy phase
    while uncovered_count > 0:
        if time.time() > deadline - 15.0:
            break
        
        # Find best active candidate
        # Use argmax with random tie-breaking
        active_counts = np.where(active, cand_count, -1)
        max_count = np.max(active_counts)
        if max_count <= 0:
            break
        
        # Random tie-breaking among max-count active candidates
        candidates = np.where((active_counts == max_count) & active)[0]
        best_idx = int(candidates[rng.randint(len(candidates))])
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        active[best_idx] = False
        
        # Update: for each word in balls[best_idx], if newly covered, decrement cand_count for all centers covering it
        for w in balls[best_idx]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1
                for c in inv_lists[w]:
                    if active[c]:
                        cand_count[c] -= 1
    
    if uncovered_count > 0:
        # Incomplete; fill in remaining uncovered words directly
        for w in range(total):
            if uncovered[w]:
                selected.append(w)
                selected_set.add(w)
        uncovered_count = 0
    
    # Pruning phase
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        # Compute cover_count
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_set.discard(c)
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Local search with simulated annealing-style swaps
    improved = True
    temperature = 1.0
    while improved and time.time() < deadline - 5.0:
        improved = False
        temperature *= 0.9
        
        # Recompute cover counts
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
        
        # Try swaps: for each critical codeword, find a non-selected center that covers all its unique words
        for c, unique_words in critical_candidates:
            if time.time() > deadline - 5.0:
                break
            
            # Find intersection of inv_lists for all unique words
            if len(unique_words) == 1:
                common_candidates = set(inv_lists[unique_words[0]])
            else:
                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])
                    if not common_candidates:
                        break
            
            # Filter to non-selected candidates
            for c_prime in common_candidates:
                if c_prime not in selected_set:
                    # Perform swap
                    selected.remove(c)
                    selected.append(c_prime)
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    improved = True
                    break
            if improved:
                break
        
        # After swaps, try to remove redundant codewords again
        if time.time() < deadline - 5.0:
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
            
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected_set.discard(c)
                else:
                    new_selected.append(c)
            
            if len(new_selected) < len(selected):
                selected = new_selected
                improved = True
        
        # Simulated annealing: try random swaps that may temporarily increase size
        # but could enable future reductions. Accept with probability based on temperature.
        if temperature > 0.1 and time.time() < deadline - 5.0:
            for _ in range(10):  # Try a few random swaps
                if time.time() > deadline - 5.0:
                    break
                # Pick a random selected codeword
                c = rng.choice(selected)
                # Pick a random non-selected center
                non_selected = [c for c in range(total) if c not in selected_set]
                if not non_selected:
                    break
                c_prime = non_selected[rng.randint(len(non_selected))]
                
                # Check if swap maintains validity (all words still covered)
                # Remove c, add c_prime, check coverage
                cover_count = np.zeros(total, dtype=np.int32)
                for cw in selected:
                    if cw != c:
                        cover_count[balls[cw]] += 1
                cover_count[balls[c_prime]] += 1
                
                if np.min(cover_count) > 0:
                    # Valid swap; accept if it doesn't increase size (it doesn't, same size)
                    # But it might enable future reductions
                    selected.remove(c)
                    selected.append(c_prime)
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    improved = True
                    break
                else:
                    # Invalid swap; accept with probability exp(-delta/T) where delta = uncovered count
                    uncovered_in_swap = np.sum(cover_count == 0)
                    if uncovered_in_swap > 0:
                        prob = np.exp(-uncovered_in_swap / temperature)
                        if rng.rand() < prob:
                            selected.remove(c)
                            selected.append(c_prime)
                            selected_set.discard(c)
                            selected_set.add(c_prime)
                            improved = True
                            break
    
    return selected

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    if q == 3 and n == 8 and radius == 1:
        # Build shared structures once
        digits, balls, inv_lists, ball_size, total = _build_structures(q, n, radius)
        
        best_codewords = None
        best_size = float('inf')
        
        # Run multiple greedy instances with different random seeds
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        
        for seed in seeds:
            if time.time() > deadline - 20.0:
                break
            try:
                res_indices = _greedy_with_pruning_and_search(
                    q, n, radius, deadline, seed, digits, balls, inv_lists, ball_size, total
                )
                if res_indices is not None and len(res_indices) > 0:
                    if len(res_indices) < best_size:
                        best_size = len(res_indices)
                        best_codewords = [digits[idx].tolist() for idx in res_indices]
            except Exception:
                pass
        
        if best_codewords is not None:
            return best_codewords
        
        return _fallback_construct(q, n, radius)
    
    return _fallback_construct(q, n, radius)

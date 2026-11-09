import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts as bitmasks for O(1) compatibility checks
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute reciprocal values
    recip = [1.0 / numbers[i] for i in range(n)]
    
    def elapsed():
        return time.time() - start_time
    
    def greedy_order(order, selected_mask, current_sum):
        """Add elements in given order if compatible. Returns (mask, sum)."""
        for i in order:
            if not (conflict_mask[i] & selected_mask):
                selected_mask |= (1 << i)
                current_sum += recip[i]
        return selected_mask, current_sum
        
    def fill_set(selected_mask, current_sum):
        """Add any compatible element if possible. Returns (mask, sum)."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for i in range(n):
                if not (selected_mask & (1 << i)):
                    if not (conflict_mask[i] & selected_mask):
                        selected_mask |= (1 << i)
                        current_sum += recip[i]
                        improved = True
        return selected_mask, current_sum

    def local_search(selected_mask, current_sum):
        """Local search with 1-swap, 2-swap, and fill. Returns (mask, sum)."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            
            # Try single swap: remove i, add best j
            for i in range(n):
                if not (selected_mask & (1 << i)):
                    continue
                new_mask = selected_mask & ~(1 << i)
                new_sum = current_sum - recip[i]
                
                # Find best j to add
                best_j = -1
                best_gain = 0.0
                
                for j in range(n):
                    if new_mask & (1 << j):
                        continue
                    if conflict_mask[j] & new_mask:
                        continue
                    gain = recip[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                
                if best_j >= 0 and new_sum + best_gain > current_sum + 1e-12:
                    selected_mask = new_mask | (1 << best_j)
                    current_sum = new_sum + best_gain
                    improved = True
                    break
            
            # Try 2-swap: remove i1, i2, add j1, j2
            if not improved and elapsed() < DEADLINE - 5:
                best_2swap_gain = 0.0
                best_i1 = best_i2 = -1
                best_j1 = best_j2 = -1
                
                # Iterate over pairs of selected elements to remove
                selected_indices = [i for i in range(n) if selected_mask & (1 << i)]
                for idx1 in range(len(selected_indices)):
                    for idx2 in range(idx1 + 1, len(selected_indices)):
                        i1 = selected_indices[idx1]
                        i2 = selected_indices[idx2]
                        new_mask = selected_mask & ~(1 << i1) & ~(1 << i2)
                        new_sum = current_sum - recip[i1] - recip[i2]
                        
                        # Find best pair to add
                        best_pair_gain = 0.0
                        best_p1 = best_p2 = -1
                        
                        for j1 in range(n):
                            if new_mask & (1 << j1):
                                continue
                            if conflict_mask[j1] & new_mask:
                                continue
                            for j2 in range(n):
                                if j2 == j1:
                                    continue
                                if new_mask & (1 << j2):
                                    continue
                                if conflict_mask[j2] & (new_mask | (1 << j1)):
                                    continue
                                if conflict_mask[j1] & (1 << j2):
                                    continue
                                pair_gain = recip[j1] + recip[j2]
                                if pair_gain > best_pair_gain:
                                    best_pair_gain = pair_gain
                                    best_p1 = j1
                                    best_p2 = j2
                        
                        total_gain = best_pair_gain - recip[i1] - recip[i2]
                        if total_gain > best_2swap_gain:
                            best_2swap_gain = total_gain
                            best_i1 = i1
                            best_i2 = i2
                            best_j1 = best_p1
                            best_j2 = best_p2
                
                if best_2swap_gain > 1e-12 and best_j1 >= 0:
                    selected_mask = (selected_mask & ~(1 << best_i1) & ~(1 << best_i2)) | (1 << best_j1) | (1 << best_j2)
                    current_sum += best_2swap_gain
                    improved = True
            
            # If no swap improved, try fill
            if not improved:
                new_mask, new_sum = fill_set(selected_mask, current_sum)
                if new_sum > current_sum + 1e-12:
                    selected_mask = new_mask
                    current_sum = new_sum
                    improved = True

        return selected_mask, current_sum
    
    # Multiple greedy orderings + local search
    best_mask = 0
    best_sum = 0.0
    
    # Ordering 1: ascending
    order1 = list(range(n))
    m, s = greedy_order(order1, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Ordering 2: descending
    order2 = list(range(n - 1, -1, -1))
    m, s = greedy_order(order2, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Ordering 3: by 1/a / degree
    order3 = sorted(range(n), key=lambda i: recip[i] / (bin(conflict_mask[i]).count('1') + 1), reverse=True)
    m, s = greedy_order(order3, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Ordering 4: by 1/a * (n - degree)
    order4 = sorted(range(n), key=lambda i: recip[i] * (n - bin(conflict_mask[i]).count('1')), reverse=True)
    m, s = greedy_order(order4, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Ordering 5: prioritize numbers with no conflicts
    no_conflict = [i for i in range(n) if conflict_mask[i] == 0]
    has_conflict = [i for i in range(n) if conflict_mask[i] != 0]
    order5 = no_conflict + sorted(has_conflict, key=lambda i: recip[i], reverse=True)
    m, s = greedy_order(order5, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # Randomized restarts
    random.seed(42)
    for _ in range(1000):
        if elapsed() >= DEADLINE - 12:
            break
        # Random order with bias toward high reciprocals
        weights = [recip[i] ** 2 for i in range(n)]
        order = sorted(range(n), key=lambda i: random.random() / (weights[i] + 1e-10))
        m, s = greedy_order(order, 0, 0.0)
        m, s = local_search(m, s)
        if s > best_sum:
            best_sum = s
            best_mask = m
            
    # Perturbation
    if elapsed() < DEADLINE - 20:
        for _ in range(200):
            if elapsed() >= DEADLINE - 8:
                break
            perturbed = best_mask
            # Remove 2-5 random elements
            k = min(random.randint(2, 5), bin(best_mask).count('1'))
            if k == 0: continue
            to_remove = random.sample(range(n), k)
            for i in to_remove:
                if best_mask & (1 << i):
                    perturbed &= ~(1 << i)
            
            # Re-add greedily in reciprocal order
            candidates = [i for i in range(n) if not (perturbed & (1 << i))]
            candidates.sort(key=lambda i: recip[i], reverse=True)
            for j in candidates:
                if not (conflict_mask[j] & perturbed):
                    perturbed |= (1 << j)
            
            pert_sum = sum(recip[i] for i in range(n) if perturbed & (1 << i))
            perturbed, pert_sum = local_search(perturbed, pert_sum)
            if pert_sum > best_sum:
                best_sum = pert_sum
                best_mask = perturbed
                
    # Final Polish
    best_mask, best_sum = fill_set(best_mask, best_sum)
    best_mask, best_sum = local_search(best_mask, best_sum)
    
    # Extra safety: verify result is valid
    result = [numbers[i] for i in range(n) if (best_mask & (1 << i))]
    
    # Verify LCM constraint
    valid = True
    for a_idx in range(len(result)):
        for b_idx in range(a_idx + 1, len(result)):
            a, b = result[a_idx], result[b_idx]
            l = a * b // gcd(a, b)
            if l <= 256:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: return just the largest number
        return [256]
    
    return result

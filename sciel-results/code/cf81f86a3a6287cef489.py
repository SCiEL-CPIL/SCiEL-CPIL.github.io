import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 50.0
    
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
    
    def greedy_order(order, start_mask=0, start_sum=0.0):
        mask = start_mask
        s = start_sum
        for i in order:
            if not (conflict_mask[i] & mask):
                mask |= (1 << i)
                s += recip[i]
        return mask, s
    
    def local_search(mask, s):
        while elapsed() < DEADLINE - 2.0:
            improved = False
            
            # Single swap: remove i, add j
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                new_mask = mask & ~(1 << i)
                new_s = s - recip[i]
                for j in range(n):
                    if new_mask & (1 << j):
                        continue
                    if not (conflict_mask[j] & new_mask):
                        cand_s = new_s + recip[j]
                        if cand_s > s + 1e-12:
                            mask = new_mask | (1 << j)
                            s = cand_s
                            improved = True
                            break
                if improved:
                    break
            
            # Multi-add: remove i, add all compatible in descending reciprocal order
            if not improved and elapsed() < DEADLINE - 2.0:
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    new_mask = mask & ~(1 << i)
                    new_s = s - recip[i]
                    candidates = []
                    for j in range(n):
                        if not (new_mask & (1 << j)) and not (conflict_mask[j] & new_mask):
                            candidates.append(j)
                    candidates.sort(key=lambda j: recip[j], reverse=True)
                    for j in candidates:
                        if not (conflict_mask[j] & new_mask):
                            new_mask |= (1 << j)
                            new_s += recip[j]
                    if new_s > s + 1e-12:
                        mask = new_mask
                        s = new_s
                        improved = True
                        break
            
            # 2-swap: remove two, add two (coordinated move)
            if not improved and elapsed() < DEADLINE - 2.0:
                # Try a few random 2-swap moves
                for _ in range(20):
                    if elapsed() >= DEADLINE - 2.0:
                        break
                    if mask.bit_count() < 2:
                        break
                    # Pick two random elements to remove
                    bits = [i for i in range(n) if mask & (1 << i)]
                    if len(bits) < 2:
                        break
                    i1, i2 = random.sample(bits, 2)
                    new_mask = mask & ~(1 << i1) & ~(1 << i2)
                    new_s = s - recip[i1] - recip[i2]
                    # Find all compatible additions
                    candidates = []
                    for j in range(n):
                        if not (new_mask & (1 << j)) and not (conflict_mask[j] & new_mask):
                            candidates.append(j)
                    candidates.sort(key=lambda j: recip[j], reverse=True)
                    # Try adding top candidates
                    added = 0
                    temp_mask = new_mask
                    temp_s = new_s
                    for j in candidates:
                        if added >= 2:
                            break
                        if not (conflict_mask[j] & temp_mask):
                            temp_mask |= (1 << j)
                            temp_s += recip[j]
                            added += 1
                    if temp_s > s + 1e-12:
                        mask = temp_mask
                        s = temp_s
                        improved = True
                        break
        
        return mask, s
    
    # Phase 1: Deterministic greedy with multiple orderings
    best_mask = 0
    best_sum = 0.0
    
    # Ordering 1: ascending a
    order1 = sorted(range(n), key=lambda i: numbers[i])
    m, s = greedy_order(order1)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 2: descending a
    order2 = sorted(range(n), key=lambda i: -numbers[i])
    m, s = greedy_order(order2)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 3: by reciprocal / degree
    order3 = sorted(range(n), key=lambda i: recip[i] / (bin(conflict_mask[i]).count('1') + 1), reverse=True)
    m, s = greedy_order(order3)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 4: by reciprocal only
    order4 = sorted(range(n), key=lambda i: recip[i], reverse=True)
    m, s = greedy_order(order4)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Phase 2: Randomized restarts
    random.seed(42)
    weights = [recip[i] ** 2 for i in range(n)]
    total_weight = sum(weights)
    
    while elapsed() < DEADLINE - 5.0:
        # Weighted random ordering
        order = []
        remaining = list(range(n))
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = random.random() * total_w
            cumulative = 0
            for idx, i in enumerate(remaining):
                cumulative += weights[i]
                if r <= cumulative:
                    order.append(i)
                    remaining.pop(idx)
                    break
            else:
                order.append(remaining.pop())
        
        m, s = greedy_order(order)
        m, s = local_search(m, s)
        if s > best_sum:
            best_sum = s
            best_mask = m
    
    # Phase 3: Perturbation from best solution
    if elapsed() < DEADLINE - 3.0:
        for _ in range(100):
            if elapsed() >= DEADLINE - 2.0:
                break
            perturbed = best_mask
            # Remove 1-3 random elements
            count = min(random.randint(1, 3), perturbed.bit_count())
            bits = [i for i in range(n) if perturbed & (1 << i)]
            to_remove = random.sample(bits, count)
            for i in to_remove:
                perturbed &= ~(1 << i)
            # Re-add greedily in reciprocal order
            candidates = [i for i in range(n) if not (perturbed & (1 << i))]
            candidates.sort(key=lambda i: recip[i], reverse=True)
            for j in candidates:
                if not (conflict_mask[j] & perturbed):
                    perturbed |= (1 << j)
            # Compute sum
            pert_sum = sum(recip[i] for i in range(n) if perturbed & (1 << i))
            # Local search on perturbed
            perturbed, pert_sum = local_search(perturbed, pert_sum)
            if pert_sum > best_sum:
                best_sum = pert_sum
                best_mask = perturbed
    
    # Convert mask back to list of numbers
    result = [numbers[i] for i in range(n) if best_mask & (1 << i)]
    return result

import numpy as np
import time
import random

N = 256

def _precompute():
    """Precompute all 4-APs and 3-APs as index arrays."""
    # 4-APs: (a, a+d, a+2d, a+3d) with 1<=a, a+3d<=256, d>=1
    four_aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    four_aps = np.array(four_aps, dtype=np.int32) - 1  # 0-indexed

    # 3-APs: (a, a+d, a+2d) with 1<=a, a+2d<=256, d>=1
    three_aps = []
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    three_aps = np.array(three_aps, dtype=np.int32) - 1  # 0-indexed

    return four_aps, three_aps

def _is_4ap_free(bitset):
    """Check if a bitset (numpy array of 0/1, length 256) has no 4-AP."""
    # For each 4-AP, check if all 4 positions are set
    # four_aps shape: (M, 4)
    # bitset[four_aps] gives shape (M, 4)
    # Check if any row is all 1s
    return not np.all(bitset[four_aps], axis=1).any()

def _count_3aps(bitset):
    """Count number of 3-APs fully contained in bitset."""
    return int(np.all(bitset[three_aps], axis=1).sum())

def _greedy_init():
    """Greedy: add elements that don't create a 4-AP, prefer those adding most 3-APs."""
    bitset = np.zeros(N, dtype=np.int8)
    selected = set()
    
    # Precompute: for each element x, which 4-APs does it participate in?
    # And which 3-APs would be completed?
    
    for x in range(1, N + 1):
        xi = x - 1
        # Check if adding x creates a 4-AP
        # x can be a, a+d, a+2d, a+3d in a 4-AP
        creates_4ap = False
        # x as a+3d: a = x-3d, need a>=1, and a, a+d, a+2d in set
        for d in range(1, (x) // 3 + 1):
            a = x - 3 * d
            if a >= 1 and (a-1) in selected and (a+d-1) in selected and (a+2*d-1) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as a+2d: a = x-2d, a+3d = x+d
        for d in range(1, (x) // 2 + 1):
            a = x - 2 * d
            if a >= 1 and (a-1) in selected and (a+d-1) in selected and (a+3*d-1) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as a+d: a = x-d, a+2d = x+d, a+3d = x+2d
        for d in range(1, (x) // 1 + 1):
            a = x - d
            if a >= 1 and (a-1) in selected and (a+2*d-1) in selected and (a+3*d-1) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as a: a+3d = x, so d=(x-a)/3, a < x
        for d in range(1, (x) // 3 + 1):
            a = x - 3 * d
            if a >= 1 and (a+d-1) in selected and (a+2*d-1) in selected and (a+3*d-1) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        
        # Count 3-APs that would be completed by adding x
        # x as middle: (x-d, x, x+d)
        cnt = 0
        for d in range(1, min(x, N - x + 1)):
            if (x - d - 1) in selected and (x + d - 1) in selected:
                cnt += 1
        # x as first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            if (x + d - 1) in selected and (x + 2 * d - 1) in selected:
                cnt += 1
        # x as last: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            if (x - 2 * d - 1) in selected and (x - d - 1) in selected:
                cnt += 1
        
        if cnt > 0 or len(selected) < 60:
            bitset[xi] = 1
            selected.add(xi)
    
    return bitset

def _local_search(bitset, max_iters=20000):
    """Local search: try swaps to improve 3-AP count while maintaining 4-AP-free."""
    current_score = _count_3aps(bitset)
    selected = set(np.where(bitset)[0].tolist())
    
    for _ in range(max_iters):
        improved = False
        # Try removing a low-contribution element and adding a high-contribution one
        # Randomly pick a few candidates
        for _ in range(5):
            if not selected:
                break
            # Pick a random element to potentially remove
            rem = random.choice(list(selected))
            # Try adding each unselected element
            best_add = -1
            best_gain = 0
            best_new_score = current_score
            
            # Sample unselected elements
            unselected = [i for i in range(N) if i not in selected]
            if not unselected:
                break
            # Sample up to 30 unselected
            candidates = random.sample(unselected, min(30, len(unselected)))
            
            for add in candidates:
                # Check if adding 'add' (after removing 'rem') creates a 4-AP
                # Temporarily remove rem
                bitset[rem] = 0
                bitset[add] = 1
                
                if _is_4ap_free(bitset):
                    new_score = _count_3aps(bitset)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_gain = new_score - current_score
                        best_add = add
                
                # Undo
                bitset[rem] = 1
                bitset[add] = 0
            
            if best_add >= 0 and best_gain > 0:
                bitset[rem] = 0
                bitset[best_add] = 1
                selected.discard(rem)
                selected.add(best_add)
                current_score = best_new_score
                improved = True
                break
        
        if not improved:
            # Try single additions
            unselected = [i for i in range(N) if i not in selected]
            if not unselected:
                break
            candidates = random.sample(unselected, min(40, len(unselected)))
            best_add = -1
            best_gain = 0
            best_new_score = current_score
            for add in candidates:
                bitset[add] = 1
                if _is_4ap_free(bitset):
                    new_score = _count_3aps(bitset)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_gain = new_score - current_score
                        best_add = add
                bitset[add] = 0
            
            if best_add >= 0 and best_gain > 0:
                bitset[best_add] = 1
                selected.add(best_add)
                current_score = best_new_score
                improved = True
            
            if not improved:
                # Try single removals that might allow better additions later
                # Just stop
                break
    
    return bitset, current_score

def _crossover(parent1, parent2):
    """Uniform crossover."""
    child = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    # Ensure 4-AP free
    if not _is_4ap_free(child):
        # Fix by removing elements that are part of 4-APs
        for _ in range(100):
            if _is_4ap_free(child):
                break
            # Find a 4-AP and remove one of its elements
            violations = np.where(np.all(child[four_aps], axis=1))[0]
            if len(violations) == 0:
                break
            v = violations[0]
            # Remove a random element from this 4-AP
            pos = random.choice(four_aps[v].tolist())
            child[pos] = 0
    return child

def _mutation(bitset, rate=0.05):
    """Random mutation: flip some bits, then fix 4-AP violations."""
    child = bitset.copy()
    for i in range(N):
        if random.random() < rate:
            child[i] = 1 - child[i]
    # Fix 4-AP violations
    for _ in range(200):
        if _is_4ap_free(child):
            break
        violations = np.where(np.all(child[four_aps], axis=1))[0]
        if len(violations) == 0:
            break
        v = violations[0]
        pos = random.choice(four_aps[v].tolist())
        child[pos] = 0
    return child

def construct():
    global four_aps, three_aps
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    four_aps, three_aps = _precompute()
    
    # Initialize population
    random.seed(42)
    np.random.seed(42)
    
    # Start with greedy
    pop = [_greedy_init()]
    # Add some random valid individuals
    for _ in range(9):
        bitset = np.zeros(N, dtype=np.int8)
        # Random subset, then fix
        for i in range(N):
            if random.random() < 0.35:
                bitset[i] = 1
        # Fix 4-AP violations
        for _ in range(500):
            if _is_4ap_free(bitset):
                break
            violations = np.where(np.all(bitset[four_aps], axis=1))[0]
            if len(violations) == 0:
                break
            v = violations[0]
            pos = random.choice(four_aps[v].tolist())
            bitset[pos] = 0
        pop.append(bitset)
    
    # Evaluate
    scores = [_count_3aps(b) for b in pop]
    best_idx = int(np.argmax(scores))
    best_bitset = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # Local search on best
    best_bitset, best_score = _local_search(best_bitset, max_iters=5000)
    
    elapsed = time.time() - start_time
    if elapsed > 65:
        return sorted((best_bitset + 1).tolist())
    
    # Genetic algorithm
    pop_size = 10
    gen = 0
    while time.time() - start_time < 75:
        gen += 1
        # Create new population
        new_pop = []
        
        # Elitism: keep best
        new_pop.append(best_bitset.copy())
        
        # Crossover + mutation
        while len(new_pop) < pop_size:
            p1_idx = int(np.random.choice(len(pop), p=None))
            p2_idx = int(np.random.choice(len(pop), p=None))
            if p1_idx == p2_idx:
                p2_idx = (p2_idx + 1) % len(pop)
            child = _crossover(pop[p1_idx], pop[p2_idx])
            if random.random() < 0.5:
                child = _mutation(child)
            new_pop.append(child)
        
        # Evaluate new population
        new_scores = [_count_3aps(b) for b in new_pop]
        
        # Update best
        max_idx = int(np.argmax(new_scores))
        if new_scores[max_idx] > best_score:
            best_score = new_scores[max_idx]
            best_bitset = new_pop[max_idx].copy()
        
        # Selection: keep top half + new
        pop = new_pop
        # Sort by score
        order = np.argsort(new_scores)[::-1]
        pop = [new_pop[i] for i in order[:pop_size]]
        
        # Local search on best every few generations
        if gen % 3 == 0:
            lb, ls = _local_search(best_bitset.copy(), max_iters=3000)
            if ls > best_score:
                best_score = ls
                best_bitset = lb
            # Also local search on a few others
            for i in range(min(2, len(pop))):
                lb2, ls2 = _local_search(pop[i].copy(), max_iters=2000)
                if ls2 > best_score:
                    best_score = ls2
                    best_bitset = lb2
        
        elapsed = time.time() - start_time
        if elapsed > 65:
            break
    
    # Final local search
    if time.time() - start_time < 78:
        best_bitset, best_score = _local_search(best_bitset, max_iters=10000)
    
    result = sorted((best_bitset + 1).tolist())
    return result

# Precompute at module level for speed
four_aps, three_aps = _precompute()

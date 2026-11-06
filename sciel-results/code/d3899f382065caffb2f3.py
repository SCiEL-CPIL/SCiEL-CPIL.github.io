import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_diffs(s):
    """Compute the set of all d > 0 such that a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                diffs.add(d)
    return diffs

def _score(s):
    """Score = number of distinct d > 0 with a 3-term AP."""
    return len(_compute_diffs(s))

def _score_with_diffs(s):
    """Return (score, diffs_set) for a set."""
    diffs = _compute_diffs(s)
    return len(diffs), diffs

def _greedy_coverage(deadline):
    """
    Maximum Coverage greedy heuristic:
    Start with an empty set, greedily add the element that maximizes
    the number of NEW distinct differences covered.
    """
    current_set = set()
    current_diffs = set()
    remaining = set(range(1, 513))
    
    while len(current_set) < 32:
        if time.time() > deadline - 10:
            break
        
        best_elem = None
        best_new_diffs = -1
        best_diffs_after = None
        
        # Sample a subset of candidates for speed
        if len(remaining) > 200:
            candidates = random.sample(list(remaining), 200)
        else:
            candidates = list(remaining)
        
        for elem in candidates:
            trial_set = current_set | {elem}
            trial_diffs = _compute_diffs(trial_set)
            new_diffs = len(trial_diffs - current_diffs)
            
            if new_diffs > best_new_diffs:
                best_new_diffs = new_diffs
                best_elem = elem
                best_diffs_after = trial_diffs
        
        if best_elem is None:
            # Fallback: pick random
            best_elem = random.choice(list(remaining))
            trial_set = current_set | {best_elem}
            best_diffs_after = _compute_diffs(trial_set)
        
        current_set.add(best_elem)
        current_diffs = best_diffs_after
        remaining.discard(best_elem)
    
    return current_set, len(current_diffs)

def _hill_climb(start_set, deadline, T0=5.0, cooling=0.995, min_T=0.05, max_stuck=300):
    """
    Simulated annealing hill climbing from a starting set.
    Returns (best_set, best_score).
    """
    current_set = set(start_set)
    current_score, current_diffs = _score_with_diffs(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    T = T0
    stuck_count = 0
    
    while time.time() < deadline - 2:
        # Pick an element to remove
        elem_out = random.choice(list(current_set))
        
        # Try several candidates to add
        best_move = None
        best_move_score = -1
        
        # Sample candidates
        available = [x for x in range(1, 513) if x not in current_set]
        if len(available) > 50:
            candidates = random.sample(available, 50)
        else:
            candidates = available
        
        for elem_in in candidates:
            new_set = set(current_set)
            new_set.discard(elem_out)
            new_set.add(elem_in)
            
            if len(new_set) != 32:
                continue
            
            new_score, _ = _score_with_diffs(new_set)
            
            if new_score > best_move_score:
                best_move_score = new_score
                best_move = (elem_out, elem_in, new_set, new_score)
        
        if best_move is None:
            stuck_count += 1
            if stuck_count > max_stuck:
                break
            continue
        
        elem_out, elem_in, new_set, new_score = best_move
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            current_score = new_score
            stuck_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_set = set(current_set)
        else:
            # Accept with probability exp(delta/T)
            if delta < -10:
                stuck_count += 1
            elif random.random() < np.exp(delta / T):
                current_set = new_set
                current_score = new_score
                stuck_count = 0
            else:
                stuck_count += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        if stuck_count > max_stuck:
            # Perturbation: make 2 random moves
            for _ in range(2):
                e_out = random.choice(list(current_set))
                avail = [x for x in range(1, 513) if x not in current_set]
                if avail:
                    e_in = random.choice(avail)
                    current_set = set(current_set)
                    current_set.discard(e_out)
                    current_set.add(e_in)
                    current_score, _ = _score_with_diffs(current_set)
            stuck_count = 0
            T = max(T, 5.0)
    
    return best_set, best_score

def _final_polish(best_set, deadline):
    """
    Final high-precision polish: try all possible single-element swaps
    and accept only improvements. Repeat until no improvement or time runs out.
    """
    current_set = set(best_set)
    current_score, _ = _score_with_diffs(current_set)
    best_set_local = set(current_set)
    best_score_local = current_score
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # For each element in the set, try replacing it with every possible candidate
        for elem_out in list(current_set):
            if time.time() > deadline - 1:
                break
            
            available = [x for x in range(1, 513) if x not in current_set]
            
            # Sample if too many
            if len(available) > 100:
                candidates = random.sample(available, 100)
            else:
                candidates = available
            
            for elem_in in candidates:
                new_set = set(current_set)
                new_set.discard(elem_out)
                new_set.add(elem_in)
                
                if len(new_set) != 32:
                    continue
                
                new_score, _ = _score_with_diffs(new_set)
                
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    
                    if new_score > best_score_local:
                        best_score_local = new_score
                        best_set_local = set(current_set)
                    break  # Move on to next elem_out
    
    return best_set_local, best_score_local

def _generate_initial_set():
    """Generate a good initial set using multiple strategies."""
    strategies = []
    
    # Strategy 1: Pure arithmetic progression
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]:
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            s = set(a + i * d for i in range(32))
            if all(1 <= x <= 512 for x in s):
                strategies.append(s)
    
    # Strategy 2: Two interleaved APs
    for _ in range(10):
        d1 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        d2 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
        s = set()
        a1 = random.randint(1, max(1, 512 - 15 * d1))
        for i in range(16):
            s.add(a1 + i * d1)
        a2 = random.randint(1, max(1, 512 - 15 * d2))
        for i in range(16):
            val = a2 + i * d2
            if val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 3: Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Strategy 4: Dense block
    base = random.randint(1, 481)
    s = set(range(base, base + 32))
    if max(s) <= 512:
        strategies.append(s)
    
    # Strategy 5: Multi-scale structure
    for _ in range(5):
        s = set()
        num_blocks = random.randint(2, 4)
        block_size = 32 // num_blocks
        for b in range(num_blocks):
            block_base = random.randint(1, 512 - block_size * 10)
            for k in range(block_size):
                val = block_base + k * random.choice([1, 2, 3])
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 6: Quadratic-like structure
    for _ in range(3):
        s = set()
        a = random.randint(1, 3)
        b = random.randint(1, 10)
        c = random.randint(1, 50)
        for i in range(32):
            val = a * i * i + b * i + c
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Pick the best
    best = None
    best_sc = -1
    for s in strategies:
        if len(s) != 32:
            continue
        sc = _score(s)
        if sc > best_sc:
            best_sc = sc
            best = s
    
    return best if best else set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    best_set = None
    best_score = 0
    
    # Phase 1: Greedy coverage from multiple random starts
    for restart in range(3):
        if time.time() > deadline - 40:
            break
        
        # Greedy construction
        greedy_set, greedy_score = _greedy_coverage(deadline)
        
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = set(greedy_set)
        
        # Hill climb from greedy result
        hc_set, hc_score = _hill_climb(greedy_set, deadline, T0=5.0, cooling=0.995, min_T=0.05, max_stuck=300)
        
        if hc_score > best_score:
            best_score = hc_score
            best_set = set(hc_set)
    
    # Phase 2: Hill climbing from diverse initial sets
    for restart in range(4):
        if time.time() > deadline - 15:
            break
        
        current_set = _generate_initial_set()
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Hill climb
        hc_set, hc_score = _hill_climb(current_set, deadline, T0=8.0, cooling=0.995, min_T=0.05, max_stuck=400)
        
        if hc_score > best_score:
            best_score = hc_score
            best_set = set(hc_set)
    
    # Phase 3: Final polish on best solution
    if best_set is not None and time.time() < deadline - 3:
        polished_set, polished_score = _final_polish(best_set, deadline)
        
        if polished_score > best_score:
            best_score = polished_score
            best_set = set(polished_set)
    
    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END

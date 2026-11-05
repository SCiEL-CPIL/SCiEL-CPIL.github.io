import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to calculate score
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)
    
    # Pre-compute prime factorization info for fast gcd checks
    # and conflict detection
    def is_coprime_to_all(a, arr):
        for b in arr:
            if gcd(a, b) != 1:
                return False
        return True
    
    # 1. Multiple Greedy Initializations
    best_set = []
    best_score = 0.0
    
    # Greedy 1: Largest first
    initial_set = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in initial_set):
            initial_set.append(a)
    if get_score(initial_set) > best_score:
        best_score = get_score(initial_set)
        best_set = list(initial_set)
    
    # Greedy 2: By score (1/(512-a)) descending, i.e., largest a first (same as above)
    # Greedy 3: Random restarts with different orderings
    import random
    random.seed(42)
    
    for _ in range(5):
        shuffled = list(range(511, 0, -1))
        random.shuffle(shuffled)
        # Sort by score descending but with some randomness
        shuffled.sort(key=lambda x: 1.0/(512-x) + random.random()*0.001, reverse=True)
        init = []
        for a in shuffled:
            if all(gcd(a, b) == 1 for b in init):
                init.append(a)
        sc = get_score(init)
        if sc > best_score:
            best_score = sc
            best_set = list(init)
    
    # 2. Enhanced Local Search
    improved = True
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        current_set = list(best_set)
        current_score = best_score
        
        # Build conflict structure
        # For each element in current_set, find which candidates it blocks
        in_set = set(current_set)
        
        # Try single removal + addition
        for i, P in enumerate(current_set):
            if time.time() > deadline:
                break
            
            remaining = current_set[:i] + current_set[i+1:]
            remaining_set = set(remaining)
            
            # Find all valid candidates (coprime to all in remaining)
            candidates = []
            for a in range(511, 0, -1):
                if a in remaining_set:
                    continue
                if is_coprime_to_all(a, remaining):
                    candidates.append(a)
            
            if not candidates:
                continue
            
            # Greedily pick best pairwise coprime subset
            new_additions = []
            for a in candidates:
                if all(gcd(a, b) == 1 for b in new_additions):
                    new_additions.append(a)
            
            delta = -1.0 / (512 - P) + sum(1.0 / (512 - a) for a in new_additions)
            
            if delta > 1e-9:
                new_set = remaining + new_additions
                new_score = current_score + delta
                if new_score > best_score:
                    best_set = list(new_set)
                    best_score = new_score
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try double removal + addition
        if len(current_set) >= 2:
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current_set)):
                    if time.time() > deadline:
                        break
                    
                    P1 = current_set[i]
                    P2 = current_set[j]
                    remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                    remaining_set = set(remaining)
                    
                    candidates = []
                    for a in range(511, 0, -1):
                        if a in remaining_set:
                            continue
                        if is_coprime_to_all(a, remaining):
                            candidates.append(a)
                    
                    if not candidates:
                        continue
                    
                    new_additions = []
                    for a in candidates:
                        if all(gcd(a, b) == 1 for b in new_additions):
                            new_additions.append(a)
                    
                    delta = (-1.0 / (512 - P1) - 1.0 / (512 - P2) + 
                             sum(1.0 / (512 - a) for a in new_additions))
                    
                    if delta > 1e-9:
                        new_set = remaining + new_additions
                        new_score = current_score + delta
                        if new_score > best_score:
                            best_set = list(new_set)
                            best_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            continue
            
        # Try add-only moves (if there's room)
        in_set = set(current_set)
        for a in range(511, 0, -1):
            if time.time() > deadline:
                break
            if a in in_set:
                continue
            if is_coprime_to_all(a, current_set):
                new_score = current_score + 1.0 / (512 - a)
                if new_score > best_score:
                    best_set = current_set + [a]
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try swap moves: remove one, add one different
            for i, P in enumerate(current_set):
                if time.time() > deadline:
                    break
                remaining = current_set[:i] + current_set[i+1:]
                remaining_set = set(remaining)
                
                for a in range(511, 0, -1):
                    if a in remaining_set:
                        continue
                    if is_coprime_to_all(a, remaining):
                        delta = -1.0 / (512 - P) + 1.0 / (512 - a)
                        if delta > 1e-9:
                            new_set = remaining + [a]
                            new_score = current_score + delta
                            if new_score > best_score:
                                best_set = list(new_set)
                                best_score = new_score
                                improved = True
                                break
                if improved:
                    break
        
        if not improved:
            break

    return best_set

import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    triple_set = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    triple_set.add((a,b,c))
    
    # For each element, list of pairs (u, w) such that (elem, u, w) is a bad triple
    conflicts = defaultdict(list)
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Precompute for each vertex, which triples it hits
    vertex_to_triples = defaultdict(list)
    for t in bad_triples:
        for v in t:
            vertex_to_triples[v].append(t)
    
    # Function to check if adding x to current set S is valid
    def can_add(S, x):
        for u, w in conflicts[x]:
            if u in S and w in S:
                return False
        return True
    
    # Branch and bound: upper bound on how many more we can add
    def upper_bound(S):
        """Count vertices not in S that can be added without conflict with S"""
        count = 0
        for x in range(1, n+1):
            if x not in S and can_add(S, x):
                count += 1
        return count
    
    best_size = 0
    best_set = set()
    
    # Run multiple randomized greedy initializations
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        random.seed(12345 + restart)
        
        current_set = set(range(1, n+1))
        active_triples = set(bad_triples)
        
        # Greedy removal with randomization
        while active_triples:
            if time.time() > deadline - 15:
                break
            best_v = None
            best_count = 0
            candidates = list(vertex_to_triples.keys())
            random.shuffle(candidates)
            
            for v in candidates:
                if v not in current_set:
                    continue
                count = 0
                for t in vertex_to_triples[v]:
                    if t in active_triples:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
                
            current_set.remove(best_v)
            to_remove = set()
            for t in vertex_to_triples[best_v]:
                if t in active_triples:
                    to_remove.add(t)
            active_triples -= to_remove
        
        # Local Search: Try to add vertices
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, n+1):
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
        
        # Phase 2: Kicks (remove 1, add many) with proper revert
        for _ in range(150):
            if time.time() > deadline - 10:
                break
            if not current_set:
                break
            
            # 1-out kick with proper revert logic
            y = random.choice(list(current_set))
            old_size = len(current_set)
            
            # Save state for potential revert
            state_before = current_set.copy()
            
            current_set.remove(y)
            
            added = []
            for x in range(1, n+1):
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    added.append(x)
            
            new_size = len(current_set)
            if new_size <= old_size:
                # Revert to state before the kick
                current_set = state_before
            # If new_size > old_size, keep the improvement and continue
        
        # Phase 3: 2-swap moves (remove y, z, add as many as possible)
        if time.time() < deadline - 12:
            for _ in range(30):
                if time.time() > deadline - 12:
                    break
                if len(current_set) < 2:
                    break
                
                # Pick two random elements to remove
                candidates_to_remove = list(current_set)
                if len(candidates_to_remove) < 2:
                    break
                y, z = random.sample(candidates_to_remove, 2)
                
                old_size = len(current_set)
                
                # Save state for potential revert
                state_before = current_set.copy()
                
                current_set.remove(y)
                current_set.remove(z)
                
                # Try to add as many as possible
                added = []
                for x in range(1, n+1):
                    if x in current_set:
                        continue
                    if can_add(current_set, x):
                        current_set.add(x)
                        added.append(x)
                
                new_size = len(current_set)
                if new_size <= old_size:
                    # Revert to state before the swap
                    current_set = state_before
                # If new_size > old_size, keep the improvement
        
        # Phase 4: Try adding back removed vertices one by one
        if time.time() < deadline - 8:
            for x in range(1, n+1):
                if x not in current_set:
                    if can_add(current_set, x):
                        current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # If we've found a very good solution, do extra polish
            if best_size >= 115 and time.time() < deadline - 5:
                # Try all possible single removals and see if we can add more
                for y in list(best_set):
                    if time.time() > deadline - 5:
                        break
                    trial = best_set.copy()
                    trial.remove(y)
                    added_any = False
                    for x in range(1, n+1):
                        if x not in trial and can_add(trial, x):
                            trial.add(x)
                            added_any = True
                    if len(trial) > len(best_set):
                        best_set = trial
                        best_size = len(best_set)

    # Final polish on best_set
    improved = True
    while improved:
        improved = False
        for x in range(1, n+1):
            if x not in best_set and can_add(best_set, x):
                best_set.add(x)
                improved = True

    result = list(best_set)
    if not result:
        result = [1]
        
    return result

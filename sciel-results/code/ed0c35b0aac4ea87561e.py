import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 590  # safety margin
    
    d = 11
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)  # C(11,4) = 330
    
    # Precompute masks
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
    
    # Precompute pairwise conflict: two 4-sets conflict if they share >= 3 indices
    conflict = [0] * n_sets
    for i in range(n_sets):
        mi = masks[i]
        for j in range(i + 1, n_sets):
            inter = mi & masks[j]
            if bin(inter).count('1') >= 3:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Build adjacency lists for fast lookup
    adj = [set() for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            if conflict[i] & (1 << j):
                adj[i].add(j)
                adj[j].add(i)
    
    rng = random.Random(42)
    
    def greedy_init(order):
        """Greedy: pick sets in given order if no conflict with already selected."""
        selected = set()
        for s in order:
            if not (adj[s] & selected):
                selected.add(s)
        return selected
    
    def local_search(initial, time_limit=12):
        """Local search with 1-add, 1-swap, and 2-for-1 moves."""
        current = set(initial)
        best = set(current)
        best_size = len(current)
        
        start = time.time()
        no_improve = 0
        max_no_improve = 300
        
        while time.time() - start < time_limit and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Try to add any compatible vertex
            for s in range(n_sets):
                if s not in current and not (adj[s] & current):
                    current.add(s)
                    improved = True
                    break
            
            if not improved:
                # Phase 2: Try 1-swap (remove one conflicting, add one)
                swapped = False
                for s in range(n_sets):
                    if s in current:
                        continue
                    conflicts = adj[s] & current
                    if len(conflicts) == 1:
                        remove = next(iter(conflicts))
                        if not (adj[s] - {remove}) & (current - {remove}):
                            current.discard(remove)
                            current.add(s)
                            swapped = True
                            improved = True
                            break
                    elif len(conflicts) == 2:
                        # Try removing both conflicts and adding s (net -1, but may enable more)
                        # Only do this if it leads to immediate gain elsewhere
                        pass
                
                if not swapped:
                    # Phase 3: Try 1-for-2 (remove one, add two)
                    swapped2 = False
                    for s in range(n_sets):
                        if s in current:
                            continue
                        conflicts_s = adj[s] & current
                        if len(conflicts_s) > 2:
                            continue
                        for t in range(n_sets):
                            if t == s or t in current or t in adj[s]:
                                continue
                            conflicts_t = adj[t] & current
                            all_remove = conflicts_s | conflicts_t
                            if len(all_remove) > 2:
                                continue
                            rem = current - all_remove
                            if not (adj[s] & rem) and not (adj[t] & rem):
                                current -= all_remove
                                current.add(s)
                                current.add(t)
                                swapped2 = True
                                improved = True
                                break
                        if swapped2:
                            break
                
                if not swapped and not swapped2:
                    no_improve += 1
            else:
                no_improve = 0
            
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
        
        return best
    
    # Multiple restarts
    best_solution = set()
    best_size = 0
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = greedy_init(order)
        improved = local_search(selected, time_limit=10)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Final polish with more time
    if time.time() < deadline - 10 and best_solution:
        polished = local_search(best_solution, time_limit=20)
        if len(polished) > best_size:
            best_size = len(polished)
            best_solution = polished
    
    # Generate points from best_solution
    points = []
    for s in best_solution:
        i, j, k, l = all_4sets[s]
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)

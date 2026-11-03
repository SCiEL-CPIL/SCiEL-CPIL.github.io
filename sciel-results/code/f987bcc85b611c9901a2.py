import numpy as np
from itertools import combinations
import random

def construct():
    D = 11
    all_subsets = list(combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that subset i and j share >= 3 indices
    conflicts = [set() for _ in range(n_subsets)]
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            if len(set(all_subsets[i]) & set(all_subsets[j])) >= 3:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    best_count = 0
    best_selected = []
    
    # Run multiple random greedy strategies
    for _ in range(200):
        available = set(range(n_subsets))
        selected = []
        
        while available:
            # Find subset in available with min conflicts within available
            min_conf = n_subsets + 1
            candidates = []
            
            # Optimization: Instead of checking all, we can just pick a random one 
            # from the ones with low conflict if we want speed, but min-conf is better.
            # To speed up, we can sample if N is large, but N=330 is small.
            
            for s in available:
                # Count conflicts in available
                # Since conflicts[s] is a set, intersection is fast
                conf_count = len(conflicts[s] & available)
                
                if conf_count < min_conf:
                    min_conf = conf_count
                    candidates = [s]
                elif conf_count == min_conf:
                    candidates.append(s)
            
            if not candidates:
                break
                
            # Randomly pick from candidates
            best_s = random.choice(candidates)
            
            selected.append(best_s)
            available.remove(best_s)
            
            # Remove all conflicts of best_s from available
            for c in conflicts[best_s]:
                if c in available:
                    available.remove(c)
                    
        # Local Improvement: Try to add any remaining
        # (Greedy already tried to add all compatible, but maybe order matters?)
        # The greedy loop above removes conflicts immediately.
        # So `available` contains subsets that were not selected and not removed as conflicts.
        # Wait, if a subset was not selected, it means it was not the "best" at some point.
        # But it might be compatible with the *final* selected set?
        # No, if it was compatible with the current selected set at step k, it would have been a candidate.
        # If it wasn't picked, it had higher conflict count than the picked one.
        # But the picked one's removal might have removed some conflicts of the unpicked one.
        # So the unpicked one might now be compatible with the *new* available set?
        # Actually, the loop continues until `available` is empty.
        # So all remaining subsets are either conflicts of selected ones or were never compatible?
        # No, `available` only shrinks by removing selected and their conflicts.
        # So if a subset is in `available` at the end, it was never selected.
        # Why? Because at every step, we picked the min-conflict one.
        # If a subset $u$ is left, it means at every step, there was another subset $v$ with $\le$ conflicts.
        # This doesn't guarantee $u$ is incompatible with the final set.
        # Example: A, B, C. A conflicts with B. B conflicts with C. A and C no conflict.
        # Step 1: A (0 conf), B (1 conf), C (1 conf). Pick A.
        # Remove A, B (conflict). Available: {C}.
        # Step 2: C (0 conf). Pick C.
        # Selected: {A, C}.
        
        # So the greedy is complete for the path taken.
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            
    # Generate vectors
    vectors = []
    for idx in best_selected:
        subset = all_subsets[idx]
        i, j, k, l = subset
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        vectors.append(v)
                        
    return np.array(vectors, dtype=np.int64)

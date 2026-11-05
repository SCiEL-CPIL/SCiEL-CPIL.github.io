"""
Max K(2,2,2)-free triple system on 16 points via optimized greedy + local search.
Uses bitmask-based conflict checking for speed, enabling more search iterations.
"""
import time
import random
import itertools

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    random.seed(42)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute: for each triple, the list of (p1,p2,p3) configurations
    # where p1,p2,p3 are distinct points not in the triple
    # For triple (x,y,z), a K(2,2,2) is formed with pairs (x,p1),(y,p2),(z,p3)
    # if all 7 other transversals are present.
    
    # Map each triple to an index for fast lookup
    triple_to_idx = {}
    for i, t in enumerate(all_triples):
        triple_to_idx[t] = i
    
    # For each triple index, precompute the list of "blocking configurations"
    # Each config is (p1, p2, p3) and the 7 other triples (as indices) that
    # together with this triple form a K(2,2,2)
    triple_configs = []
    for t in all_triples:
        x, y, z = t
        others = [i for i in range(N) if i not in t]
        configs = []
        for p1 in others:
            for p2 in others:
                if p2 == p1:
                    continue
                for p3 in others:
                    if p3 == p1 or p3 == p2:
                        continue
                    # The 7 other transversals
                    other_triples = [
                        (x, y, p3),
                        (x, p2, z),
                        (x, p2, p3),
                        (p1, y, z),
                        (p1, y, p3),
                        (p1, p2, z),
                        (p1, p2, p3),
                    ]
                    # Convert to sorted tuples for lookup
                    other_indices = []
                    valid = True
                    for ot in other_triples:
                        ot_sorted = tuple(sorted(ot))
                        if ot_sorted in triple_to_idx:
                            other_indices.append(triple_to_idx[ot_sorted])
                        else:
                            valid = False
                            break
                    if valid:
                        configs.append(tuple(other_indices))
        triple_configs.append(configs)
    
    def check_violation_fast(triple_idx, present_set):
        """Check if adding triple at index triple_idx would create K(2,2,2).
        present_set is a set of triple indices that are currently present."""
        if triple_idx in present_set:
            return False
        for config in triple_configs[triple_idx]:
            # Check if all 7 other triples are in present_set
            all_present = True
            for idx in config:
                if idx not in present_set:
                    all_present = False
                    break
            if all_present:
                return True
        return False
    
    def init_star():
        """Create initial triple set: all triples through point 0 (105 triples)."""
        s = set()
        for a in range(1, N):
            for b in range(a+1, N):
                t = (0, a, b)
                s.add(triple_to_idx[t])
        return s
    
    # Start with star
    current = init_star()
    best = set(current)
    best_size = len(best)
    
    # Greedy phase: try to add triples
    candidates = [i for i in range(n_triples) if i not in current]
    random.shuffle(candidates)
    
    # Multiple greedy passes
    for pass_num in range(20):
        if time.time() > deadline - 10:
            break
        added_any = False
        for t_idx in candidates:
            if time.time() > deadline - 10:
                break
            if t_idx in current:
                continue
            if not check_violation_fast(t_idx, current):
                current.add(t_idx)
                added_any = True
        if not added_any:
            break
        # Rebuild candidates
        candidates = [i for i in range(n_triples) if i not in current]
        random.shuffle(candidates)
    
    if len(current) > best_size:
        best = set(current)
        best_size = len(best)
    
    # Local search phase: try remove-and-readd moves
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        
        # Try random remove-and-readd
        if len(current) == 0:
            break
        
        t_remove = random.choice(tuple(current))
        
        trial = set(current)
        trial.remove(t_remove)
        
        # Try to add up to 2-3 new triples
        # Focus on triples that might have been blocked
        blocked_candidates = [i for i in range(n_triples) if i not in trial and i != t_remove]
        random.shuffle(blocked_candidates)
        
        added_count = 0
        for t_idx in blocked_candidates[:150]:
            if time.time() > deadline - 5:
                break
            if t_idx in trial:
                continue
            if not check_violation_fast(t_idx, trial):
                trial.add(t_idx)
                added_count += 1
                if added_count >= 2:
                    break
        
        if len(trial) > len(current):
            current = trial
            if len(current) > best_size:
                best = set(current)
                best_size = len(best)
    
    # Final greedy pass
    candidates_final = [i for i in range(n_triples) if i not in current]
    random.shuffle(candidates_final)
    for t_idx in candidates_final:
        if time.time() > deadline - 2:
            break
        if not check_violation_fast(t_idx, current):
            current.add(t_idx)
    
    if len(current) > best_size:
        best = set(current)
        best_size = len(best)
    
    # Convert to list of lists
    result = [list(all_triples[i]) for i in sorted(best)]
    return result

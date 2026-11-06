import time
import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for comb in combinations(range(16), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Precompute conflicts for each pair?
    # Let's try to precompute `conflicts[i][j]` as a list of set indices that conflict with pair (i, j).
    # This might be too much memory/time to precompute fully.
    # Let's try on-the-fly filtering with a set of candidates.
    
    def get_conflicts_for_pair(A, B, candidates_indices):
        # Returns list of indices in candidates_indices that conflict with pair (A, B)
        C = A & B
        forbidden = (A | B) & (~C)
        conflicts = []
        for idx in candidates_indices:
            S = all_sets[idx]
            if (S & C) == C and (S & forbidden) == 0:
                conflicts.append(idx)
        return conflicts

    def greedy_build(F, candidates):
        # F: list of masks
        # candidates: list of indices of sets not in F
        in_F = set(F)
        
        while True:
            if not candidates:
                break
            
            # Pick a random candidate to try
            # Shuffling the whole list is expensive.
            # Pick a random index in the candidates list
            pos = random.randint(0, len(candidates) - 1)
            idx = candidates[pos]
            S = all_sets[idx]
            
            # Check if S can be added
            valid = True
            for i in range(len(F)):
                A = F[i]
                for j in range(i + 1, len(F)):
                    B = F[j]
                    C = A & B
                    if (S & C) == C:
                        forbidden = (A | B) & (~C)
                        if (S & forbidden) == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Add S to F
                F.append(S)
                in_F.add(S)
                
                # Remove S from candidates
                candidates.pop(pos)
                
                # Update candidates: Remove any that conflict with new pairs involving S
                # New pairs are (S, B) for B in F[:-1]
                # We need to filter candidates against these new pairs
                # This is O(|F| * |candidates|)
                # To avoid O(N^2) blowup, we can do this periodically or if |F| is small.
                # For now, let's just filter.
                if len(F) < 50: # Optimization: only do full filter if small
                    new_candidates = []
                    for c_idx in candidates:
                        c_S = all_sets[c_idx]
                        ok = True
                        # Check against new pairs (S, B)
                        for B in F[:-1]:
                            C = S & B
                            if (c_S & C) == C:
                                forbidden = (S | B) & (~C)
                                if (c_S & forbidden) == 0:
                                    ok = False
                                    break
                        if ok:
                            new_candidates.append(c_idx)
                    candidates = new_candidates
                else:
                    # If F is large, filtering is expensive.
                    # We can just leave candidates as is, and rely on the check in the loop.
                    # But this means we might pick invalid candidates repeatedly.
                    # A better strategy for large F:
                    # Don't filter, just check.
                    pass
            else:
                # S is invalid, remove it from candidates
                candidates.pop(pos)
                
        return F, candidates

    start_time = time.time()
    deadline = start_time + 50.0
    
    best_F = []
    best_size = 0
    
    # Initial run
    F, _ = greedy_build([], list(range(n_sets)))
    if len(F) > best_size:
        best_F = F
        best_size = len(F)
        
    # Iterated Local Search
    while time.time() < deadline:
        F = best_F[:]
        candidates = [i for i in range(n_sets) if all_sets[i] not in F]
        
        # Perturb
        k = 5
        if len(F) > k:
            to_remove = random.sample(F, k)
            for s in to_remove:
                F.remove(s)
                # Re-add to candidates
                # Find index
                for i in range(n_sets):
                    if all_sets[i] == s:
                        candidates.append(i)
                        break
        
        F, _ = greedy_build(F, candidates)
        
        if len(F) > best_size:
            best_F = F
            best_size = len(F)

    result = []
    for mask in best_F:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

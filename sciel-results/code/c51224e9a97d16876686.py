import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Step 1: Generate all 4-subsets
    subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(subsets) # 330
    
    # Precompute intersection sizes between all pairs of subsets
    # inter[i][j] = len(subsets[i] & subsets[j])
    # We only care if inter[i][j] >= 3
    # For efficiency, we can store the list of incompatible subsets for each subset
    incompatible_supports = [set() for _ in range(n_subsets)]
    for i in range(n_subsets):
        si = set(subsets[i])
        for j in range(i + 1, n_subsets):
            sj = set(subsets[j])
            if len(si & sj) >= 3:
                incompatible_supports[i].add(j)
                incompatible_supports[j].add(i)
    
    # Convert to lists for faster membership testing
    incompat_lists = [list(s) for s in incompatible_supports]
    
    # Step 2: Find a large independent set of supports
    # Greedy with randomization + local search improvement
    best_supports = []
    best_size = 0
    
    def greedy_random(order):
        """Greedy independent set given an order."""
        selected = set()
        for idx in order:
            if not (selected & incompatible_supports[idx]):
                selected.add(idx)
        return selected
    
    def local_search_improve(selected):
        """Try to improve the independent set via swap operations."""
        selected = set(selected)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try swapping each selected element with an unselected one
            for s in list(selected):
                # Find candidates: unselected, compatible with (selected - {s})
                candidates = []
                for c in range(n_subsets):
                    if c in selected:
                        continue
                    # Check if c is compatible with all in selected except s
                    if not (incompatible_supports[c] & (selected - {s})):
                        candidates.append(c)
                if candidates:
                    # Try each candidate swap
                    for c in candidates:
                        new_size = len(selected)  # swap keeps same size
                        # Actually swap: remove s, add c
                        # We want to see if we can grow: try adding c without removing s first
                        pass
                # Try pure addition: find an unselected node compatible with all selected
                for c in range(n_subsets):
                    if c not in selected:
                        if not (selected & incompatible_supports[c]):
                            selected.add(c)
                            improved = True
                            break
                if improved:
                    break
            # Try swap: remove one selected, add two unselected that are compatible
            if not improved:
                for s in list(selected):
                    if time.time() > deadline - 5:
                        return selected
                    remaining = selected - {s}
                    # Find pairs of unselected nodes compatible with remaining and each other
                    added = False
                    for c1 in range(n_subsets):
                        if c1 in selected:
                            continue
                        if incompatible_supports[c1] & remaining:
                            continue
                        for c2 in range(c1 + 1, n_subsets):
                            if c2 in selected:
                                continue
                            if incompatible_supports[c2] & remaining:
                                continue
                            if c1 in incompatible_supports[c2]:
                                continue
                            # Swap s for c1, c2
                            new_set = remaining | {c1, c2}
                            # Verify it's truly independent
                            valid = True
                            for x in new_set:
                                if incompatible_supports[x] & new_set:
                                    valid = False
                                    break
                            if valid:
                                selected = new_set
                                improved = True
                                added = True
                                break
                        if added:
                            break
                    if added:
                        break
        return selected
    
    # Multiple randomized greedy restarts with local search
    n_restarts = 200
    for seed in range(n_restarts):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        order = list(range(n_subsets))
        rng.shuffle(order)
        
        selected = greedy_random(order)
        
        # Local search improvement
        selected = local_search_improve(selected)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_supports = list(selected)
    
    # If we have time, do more restarts with different strategies
    remaining_time = deadline - 15 - time.time()
    if remaining_time > 5:
        # Try more restarts
        extra_restarts = int(remaining_time / 0.05)  # ~50ms per restart
        for seed in range(n_restarts, n_restarts + extra_restarts):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            order = list(range(n_subsets))
            rng.shuffle(order)
            
            selected = greedy_random(order)
            selected = local_search_improve(selected)
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_supports = list(selected)
    
    # Step 3: Construct vectors
    vectors = []
    
    # Add weight 4 vectors for selected supports
    # Use even-parity sign vectors (8 per block) to ensure compatibility
    # Two weight-4 vectors on disjoint supports: distance^2 = 4+4+4+4+4+4+4+4 = 32 (8 positions differ by 2)
    # Wait, let me think about this more carefully.
    # v1 has support S1, v2 has support S2. |S1 ∩ S2| <= 2.
    # On positions in S1∩S2: both have ±1, so diff is 0 or ±2
    # On positions in S1\S2: v1 has ±1, v2 has 0, so diff is ±1
    # On positions in S2\S1: v1 has 0, v2 has ±1, so diff is ±1
    # On positions in neither: both 0, diff is 0
    # ||v1||^2 = 4, ||v2||^2 = 4
    # ||v1-v2||^2 = sum of squared diffs
    # If |S1∩S2| = k, then 4-k positions in S1\S2 contribute 1 each, 4-k in S2\S1 contribute 1 each
    # k positions in intersection contribute (±1-±1)^2 which is 0 or 4
    # So ||v1-v2||^2 >= 2*(4-k) = 8-2k
    # For k=2: >= 4, which is < 4 = max norm^2. Problem!
    # For k=1: >= 6 > 4. OK
    # For k=0: >= 8 > 4. OK
    # So we need |S1∩S2| <= 1 for safety with all sign combos.
    # But the current code allows |S1∩S2| <= 2 (incompatible if >= 3).
    # Let me check: if k=2 and signs are opposite on both intersection positions:
    # diff^2 = 2*4 + 2*2*1 = 8+4 = 12 > 4. OK
    # If signs are same on both: diff^2 = 0 + 2*2*1 = 4 = max norm^2. Borderline!
    # We need >= max norm. Max norm is 2, so norm^2 = 4. Distance^2 >= 4 means distance >= 2. OK.
    # Actually the requirement is min pairwise distance >= max norm.
    # Max norm = 2 (for weight-4 vectors, norm = sqrt(4) = 2; for weight-1 vectors, norm = 2)
    # So we need ||vi - vj|| >= 2 for all i != j, i.e., ||vi-vj||^2 >= 4.
    # With k=2 and same signs on intersection: ||v1-v2||^2 = 0+0+1+1+1+1+1+1 = 6 > 4. Wait no.
    # Let me recount. S1 has 4 elements, S2 has 4 elements, |S1∩S2| = 2.
    # S1\S2 has 2 elements, S2\S1 has 2 elements.
    # On S1∩S2 (2 positions): if same sign, diff=0, contributes 0. If opposite, diff=±2, contributes 4.
    # On S1\S2 (2 positions): v1=±1, v2=0, diff=±1, contributes 1 each. Total 2.
    # On S2\S1 (2 positions): v1=0, v2=±1, diff=∓1, contributes 1 each. Total 2.
    # So ||v1-v2||^2 = (0 or 8) + 2 + 2 = 4 or 12.
    # Minimum is 4 when signs agree on intersection. 4 >= 4. OK!
    
    # So all 16 sign combos are fine as long as |S1∩S2| <= 2.
    for idx in best_supports:
        subset = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, pos in enumerate(subset):
                v[pos] = signs[i]
            vectors.append(v)
            
    # Add weight 1 vectors (±2e_i)
    for pos in range(D):
        for sign in [2, -2]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = sign
            vectors.append(v)
            
    # Convert to array
    V = np.array(vectors, dtype=np.int64)
    
    return V

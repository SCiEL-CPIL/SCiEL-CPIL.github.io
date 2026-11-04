import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)

    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)

    # Conflict matrix: conflict if intersection >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                adj[i].add(j)
                adj[j].add(i)

    # Precompute non-neighbor masks for fast MRV scoring
    non_adj_masks = []
    for i in range(N_S):
        mask = 0
        for j in range(N_S):
            if j != i and not conflict[i, j]:
                mask |= (1 << j)
        non_adj_masks.append(mask)

    rng = np.random.RandomState(42)

    def mrv_greedy_init(seed):
        """Minimum Remaining Values heuristic for independent set construction."""
        local_rng = np.random.RandomState(seed)
        remaining = set(range(N_S))
        selected = []
        
        while remaining:
            # Score each candidate by number of compatible remaining nodes
            best_score = -1
            best_candidates = []
            for v in remaining:
                # Count how many nodes in 'remaining' are compatible with v
                # A node u is compatible if u is in remaining and u is not in adj[v]
                # i.e., u is in remaining AND not in adj[v]
                # We can compute this as |remaining ∩ non_adj[v]|
                # For speed with sets:
                # compatible_count = len(remaining - adj[v])
                # But adj[v] is a set. remaining is a set.
                # This is O(|remaining|) per v, so O(|remaining|^2) total per step.
                # N_S is 330. 330^2 is ~100k. Number of steps is ~15. Total ~1.5M ops. Acceptable.
                
                # Optimization: precompute adj as sets for fast intersection
                comp_count = len(remaining) - len(remaining & adj[v])
                if comp_count > best_score:
                    best_score = comp_count
                    best_candidates = [v]
                elif comp_count == best_score:
                    best_candidates.append(v)
            
            # Pick randomly from best candidates to add diversity
            v = local_rng.choice(best_candidates)
            selected.append(v)
            remaining.remove(v)
            # Remove all neighbors of v from remaining
            for u in adj[v]:
                if u in remaining:
                    remaining.remove(u)
        
        return set(selected)

    def local_search(x, max_iter=5000, time_limit=None):
        """Local search on the independent set."""
        current_x = set(x)
        current_score = len(current_x)
        best_x = set(current_x)
        best_score = current_score
        
        # Precompute blocked for current set
        blocked = set()
        for v in current_x:
            blocked.update(adj[v])
        
        T = 10.0
        cooling = 0.999
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            r = rng.random()
            
            if r < 0.4:
                # Remove 1-2 random nodes
                if len(current_x) == 0:
                    continue
                n_remove = 1 + rng.randint(0, 1)
                to_remove = list(rng.choice(list(current_x), size=min(n_remove, len(current_x)), replace=False))
                for v in to_remove:
                    current_x.discard(v)
                
                # Re-greedy fill
                # Rebuild blocked
                blocked = set()
                for v in current_x:
                    blocked.update(adj[v])
                
                # Add any compatible nodes
                changed = True
                while changed:
                    changed = False
                    candidates = [v for v in range(N_S) if v not in current_x and v not in blocked]
                    if candidates:
                        v = candidates[rng.randint(len(candidates))]
                        current_x.add(v)
                        blocked.update(adj[v])
                        changed = True
                        
                current_score = len(current_x)
                if current_score > best_score:
                    best_score = current_score
                    best_x = set(current_x)
                    
            elif r < 0.7:
                # Swap: remove 1, add 1
                if not current_x:
                    continue
                u = rng.choice(list(current_x))
                temp_x = current_x.copy()
                temp_x.discard(u)
                
                temp_blocked = set()
                for v in temp_x:
                    temp_blocked.update(adj[v])
                
                candidates = [v for v in range(N_S) if v not in temp_x and v not in temp_blocked]
                if candidates:
                    v = candidates[rng.randint(len(candidates))]
                    current_x.discard(u)
                    current_x.add(v)
                    
                    # Update blocked
                    blocked = set()
                    for w in current_x:
                        blocked.update(adj[w])
                    
                    # Greedy fill
                    changed = True
                    while changed:
                        changed = False
                        cands = [w for w in range(N_S) if w not in current_x and w not in blocked]
                        if cands:
                            w = cands[rng.randint(len(cands))]
                            current_x.add(w)
                            blocked.update(adj[w])
                            changed = True
                    
                    current_score = len(current_x)
                    if current_score > best_score:
                        best_score = current_score
                        best_x = set(current_x)
            else:
                # 2-swap: remove 2, add 2
                if len(current_x) < 2:
                    continue
                u1, u2 = rng.choice(list(current_x), size=2, replace=False)
                temp_x = current_x.copy()
                temp_x.discard(u1)
                temp_x.discard(u2)
                
                temp_blocked = set()
                for v in temp_x:
                    temp_blocked.update(adj[v])
                
                candidates = [v for v in range(N_S) if v not in temp_x and v not in temp_blocked]
                if len(candidates) >= 2:
                    v1, v2 = rng.choice(candidates, size=2, replace=False)
                    if not conflict[v1, v2]:
                        current_x.discard(u1)
                        current_x.discard(u2)
                        current_x.add(v1)
                        current_x.add(v2)
                        
                        blocked = set()
                        for w in current_x:
                            blocked.update(adj[w])
                        
                        # Greedy fill
                        changed = True
                        while changed:
                            changed = False
                            cands = [w for w in range(N_S) if w not in current_x and w not in blocked]
                            if cands:
                                w = cands[rng.randint(len(cands))]
                                current_x.add(w)
                                blocked.update(adj[w])
                                changed = True
                        
                        current_score = len(current_x)
                        if current_score > best_score:
                            best_score = current_score
                            best_x = set(current_x)
            
            T *= cooling
            
        return best_x

    best_support_set = set()
    best_size = 0
    
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Use MRV greedy for initialization
        x0 = mrv_greedy_init(42 + r)
        
        remaining_time = deadline - time.time()
        # Allocate time per restart
        max_iter = min(5000, max(1000, int(remaining_time * 50)))
        
        x_best = local_search(x0, max_iter=max_iter, time_limit=deadline - 5.0)
        size = len(x_best)
        
        if size > best_size:
            best_size = size
            best_support_set = x_best

    # Final polish
    if time.time() < deadline - 3.0 and best_support_set:
        polish_x = set(best_support_set)
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for v in range(N_S):
                if v in polish_x:
                    continue
                compatible = all(not conflict[v, u] for u in polish_x)
                if compatible:
                    polish_x.add(v)
                    changed = True
        if len(polish_x) > best_size:
            best_size = len(polish_x)
            best_support_set = polish_x

    # Construct vectors
    vectors = []
    
    # Type A: +/- 2 e_i
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: 4 non-zeros of +/- 1 on selected supports
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    # Mixed vectors: norm sqrt(8). 
    # These are vectors with 4 non-zeros of +/- 2? No, norm would be 4.
    # The knowledge snippet says "96 'mixed' integer vectors of norm sqrt(8)".
    # Norm sqrt(8) means sum of squares = 8.
    # Integer vectors with sum of squares 8 in R^11.
    # Possible patterns:
    # 1. One 2, one 2, rest 0? 2^2 + 2^2 = 8. Support size 2.
    # 2. One 2, two 1s, rest 0? 2^2 + 1^2 + 1^2 = 6. No.
    # 3. Two 2s? 4+4=8. Yes.
    # 4. One 2, one sqrt(4)? No integers.
    # 5. Eight 1s? 8*1=8. Support size 8.
    # 6. One 2, one 2, one 1, one 1? 4+4+1+1=10. No.
    # 7. One 2, one 1, one 1, one 1, one 1? 4+1+1+1+1=8. Support size 5.
    # 8. One 2, one 2? Support size 2.
    
    # Let's check compatibility of "Support size 2, values +/-2" with existing sets.
    # Vector v with v_i=2, v_j=2. Norm^2 = 8. Norm = 2*sqrt(2) ~ 2.82.
    # This is LARGER than the max norm of Type A (2) and Type D (2).
    # The problem requires: min pairwise distance >= max vector norm.
    # If we add vectors of norm 2.82, the max norm becomes 2.82.
    # Then ALL pairwise distances must be >= 2.82.
    # Type A vectors have norm 2. Distance between two Type A is at least 2 (e.g. 2e1 and 2e2 -> dist sqrt(8)=2.82).
    # Distance 2e1 and -2e1 is 4.
    # Distance 2e1 and 2e2 is sqrt(8) = 2.82. This is >= 2.82. OK.
    # Distance 2e1 and Type D (norm 2).
    # If Type D has 0 at i, dist^2 = 4 + 4 = 8. Dist = 2.82. OK.
    # If Type D has 1 at i, dist^2 = (2-1)^2 + 3 = 4. Dist = 2. < 2.82. FAIL.
    # So we CANNOT mix Type A (norm 2) with vectors of norm > 2 if the distance drops below the new max norm.
    
    # The knowledge snippet says: "Adding the 96 'mixed' integer vectors of norm sqrt(8) ... increases the count from 252 to 348."
    # This implies the max norm might have changed, or these vectors were compatible with a configuration where max norm was sqrt(8)?
    # If max norm is sqrt(8), then Type A (norm 2) is fine, but we need dist >= sqrt(8).
    # Type A vectors: min dist is sqrt(8) (between orthogonal ones). OK.
    # Type D vectors: norm 2.
    # Distance between Type D and Type D: min dist 2. < sqrt(8). FAIL.
    # So we cannot have Type D and these mixed vectors together if max norm is sqrt(8).
    
    # Perhaps the "mixed" vectors replace Type D? Or the configuration is different?
    # "Missing from the lattice-based construction".
    # Maybe the base construction was ONLY Type A and these Mixed vectors?
    # If we use ONLY Type A (norm 2) and Mixed (norm sqrt(8)):
    # Max norm = sqrt(8).
    # Dist(Type A, Type A) >= sqrt(8). OK.
    # Dist(Mixed, Mixed) ?
    # Dist(Type A, Mixed) ?
    # Let's check Dist(2e1, 2e1+2e2) = 2e2. Norm 2. < sqrt(8). FAIL.
    # So 2e1 and 2e1+2e2 are incompatible if max norm is sqrt(8).
    
    # This suggests the "mixed" vectors might be of a different type or the knowledge snippet refers to a specific subset.
    # Given the risk and the fact that the current score is 2.64 (likely ~250 vectors), and the snippet claims 348,
    # but my analysis shows incompatibility with the current Type D set, I will NOT add these vectors blindly.
    # The current program's Type D selection is the main driver.
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return np.array(vectors, dtype=np.int8)

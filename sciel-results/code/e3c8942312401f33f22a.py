import random
import time
import math

N = 16
FULL_MASK = (1 << N) - 1

def _build_covered(triples_masks):
    """Build covered array: covered[S]=1 if S contains at least one triple."""
    covered = bytearray(1 << N)
    for t_mask in triples_masks:
        rest = FULL_MASK ^ t_mask
        sub = rest
        while True:
            covered[t_mask | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
    return covered

def _find_max_independent_set(covered):
    """Find a maximum independent set (largest S with covered[S]=0)."""
    best_set = 0
    best_size = 0
    for S in range(1 << N):
        if not covered[S]:
            c = S.bit_count()
            if c > best_size:
                best_size = c
                best_set = S
    return best_set, best_size

def _triple_mask(i, j, k):
    return (1 << i) | (1 << j) | (1 << k)

def _is_linear_with(triple, existing_triples):
    """Check if triple is linear with all existing triples."""
    a, b, c = triple
    for (x, y, z) in existing_triples:
        common = 0
        if a == x or a == y or a == z: common += 1
        if b == x or b == y or b == z: common += 1
        if c == x or c == y or c == z: common += 1
        if common > 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s
    
    rng = random.Random(123)
    
    # Precompute all possible triples
    all_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                all_triples.append((i, j, k))
    
    # Greedy initial system: add triples in random order while maintaining linearity
    def greedy_initial():
        triples = []
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        for t in shuffled:
            if _is_linear_with(t, triples):
                triples.append(t)
        return triples
    
    cur_triples = greedy_initial()
    cur_masks = [_triple_mask(*t) for t in cur_triples]
    
    # Track best
    covered = _build_covered(cur_masks)
    _, cur_alpha = _find_max_independent_set(covered)
    best_alpha = cur_alpha
    best_triples = list(cur_triples)
    
    # Precompute for each pair, the list of triples containing that pair
    pair_to_triples = {}
    for idx, (i, j, k) in enumerate(all_triples):
        for p1, p2 in [(i,j), (i,k), (j,k)]:
            key = (min(p1,p2), max(p1,p2))
            if key not in pair_to_triples:
                pair_to_triples[key] = []
            pair_to_triples[key].append(idx)
    
    # Targeted local search
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Get current covered and max independent set
        covered = _build_covered(cur_masks)
        max_ind_set, cur_alpha = _find_max_independent_set(covered)
        
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_triples = list(cur_triples)
        
        # Extract points in max independent set
        ind_points = [p for p in range(N) if (max_ind_set >> p) & 1]
        
        # Strategy 1: Try to add a triple that covers a pair within the independent set
        added = False
        if len(ind_points) >= 2:
            # Get all pairs within the independent set
            ind_pairs = []
            for i in range(len(ind_points)):
                for j in range(i + 1, len(ind_points)):
                    p1, p2 = ind_points[i], ind_points[j]
                    key = (min(p1, p2), max(p1, p2))
                    ind_pairs.append((key, p1, p2))
            
            rng.shuffle(ind_pairs)
            for key, p1, p2 in ind_pairs:
                # Find triples containing this pair
                candidates = pair_to_triples.get(key, [])
                rng.shuffle(candidates)
                for cand_idx in candidates:
                    t = all_triples[cand_idx]
                    # Check if t is already in cur_triples
                    if t in cur_triples:
                        continue
                    # Check linearity
                    if _is_linear_with(t, cur_triples):
                        # Add it
                        cur_triples.append(t)
                        cur_masks.append(_triple_mask(*t))
                        added = True
                        break
                if added:
                    break
        
        # Strategy 2: If no add, try a swap
        if not added and cur_triples:
            # Remove a random triple
            rem_idx = rng.randrange(len(cur_triples))
            rem_t = cur_triples[rem_idx]
            
            # Try to find a triple to add
            found_add = False
            for _ in range(20):
                add_idx = rng.randrange(len(all_triples))
                add_t = all_triples[add_idx]
                if add_t == rem_t:
                    continue
                # Check linearity with all triples except rem_t
                ok = True
                for j, e in enumerate(cur_triples):
                    if j == rem_idx:
                        continue
                    if not _is_linear_with(add_t, [e]):
                        ok = False
                        break
                if ok:
                    cur_triples[rem_idx] = add_t
                    cur_masks[rem_idx] = _triple_mask(*add_t)
                    found_add = True
                    break
            
            if not found_add:
                # Revert
                cur_triples[rem_idx] = rem_t
                cur_masks[rem_idx] = _triple_mask(*rem_t)
        
        # Strategy 3: Random add or remove for diversity
        if not added:
            r = rng.random()
            if r < 0.3 and cur_triples:
                # Remove
                rem_idx = rng.randrange(len(cur_triples))
                rem_t = cur_triples[rem_idx]
                rem_mask = cur_masks[rem_idx]
                cur_triples.pop(rem_idx)
                cur_masks.pop(rem_idx)
            elif r < 0.6:
                # Add random linear triple
                for _ in range(10):
                    add_idx = rng.randrange(len(all_triples))
                    add_t = all_triples[add_idx]
                    if add_t not in cur_triples and _is_linear_with(add_t, cur_triples):
                        cur_triples.append(add_t)
                        cur_masks.append(_triple_mask(*add_t))
                        break
        
        T *= cooling
    
    return [list(t) for t in best_triples]

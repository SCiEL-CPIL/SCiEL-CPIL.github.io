import time
import random

N = 256
TIME_BUDGET = 78.0  # 90s evaluator, keep safety margin

# Precompute 3-APs (positive diff, counted once)
three_aps = []
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_aps.append((a - 1, b - 1, c - 1))  # 0-indexed

# Precompute 4-APs
four_aps = []
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_aps.append((a - 1, b - 1, c - 1, e - 1))

# For fast checking: which 4-APs does each element belong to?
elem_to_4aps = {i: [] for i in range(N)}
for idx, ap in enumerate(four_aps):
    for elem in ap:
        elem_to_4aps[elem].append(idx)

# For fast checking: which 3-APs does each element belong to?
elem_to_3aps = {i: [] for i in range(N)}
for idx, ap in enumerate(three_aps):
    for elem in ap:
        elem_to_3aps[elem].append(idx)

# Convert 4-APs to bitmask representations for fast "is blocked" check
# A 4-AP (a,b,c,e) blocks an element x if the other 3 elements are present.
# We precompute for each 4-AP a mask of the 4 elements.
# To check if adding x is allowed: for each 4-AP containing x, check if
# the other 3 elements are all in the set.
# Represent set as bitmask (int).
# Precompute for each 4-AP and for each position in it: the mask of the OTHER 3 elements
# elem_pos_to_other_mask[ap_idx][pos] = mask of the other 3 elements
elem_pos_to_other_mask = [[0] * 4 for _ in range(len(four_aps))]
for idx, ap in enumerate(four_aps):
    for pos in range(4):
        mask = 0
        for q in range(4):
            if q != pos:
                mask |= (1 << ap[q])
        elem_pos_to_other_mask[idx][pos] = mask

# For each element, store list of (ap_idx, pos) so we can check
elem_to_4ap_pos = {i: [] for i in range(N)}
for idx, ap in enumerate(four_aps):
    for pos in range(4):
        elem_to_4ap_pos[ap[pos]].append((idx, pos))

# For 3-APs: for each 3-AP and position, mask of other 2 elements
elem_pos_to_3ap_other_mask = [[0] * 3 for _ in range(len(three_aps))]
for idx, ap in enumerate(three_aps):
    for pos in range(3):
        mask = 0
        for q in range(3):
            if q != pos:
                mask |= (1 << ap[q])
        elem_pos_to_3ap_other_mask[idx][pos] = mask

elem_to_3ap_pos = {i: [] for i in range(N)}
for idx, ap in enumerate(three_aps):
    for pos in range(3):
        elem_to_3ap_pos[ap[pos]].append((idx, pos))


def is_4ap_free_add(mask, x):
    """Check if adding x to set (bitmask) keeps it 4-AP-free."""
    for ap_idx, pos in elem_to_4ap_pos[x]:
        other_mask = elem_pos_to_other_mask[ap_idx][pos]
        if (mask & other_mask) == other_mask:
            return False
    return True


def count_3aps_in_mask(mask):
    """Count 3-APs fully present in mask."""
    count = 0
    for ap_idx, (a, b, c) in enumerate(three_aps):
        m = (1 << a) | (1 << b) | (1 << c)
        if (mask & m) == m:
            count += 1
    return count


def delta_3aps_add(mask, x):
    """Number of new 3-APs created by adding x to mask."""
    count = 0
    for ap_idx, pos in elem_to_3ap_pos[x]:
        other_mask = elem_pos_to_3ap_other_mask[ap_idx][pos]
        if (mask & other_mask) == other_mask:
            count += 1
    return count


def delta_3aps_remove(mask, x):
    """Number of 3-APs destroyed by removing x from mask."""
    count = 0
    for ap_idx, pos in elem_to_3ap_pos[x]:
        other_mask = elem_pos_to_3ap_other_mask[ap_idx][pos]
        if (mask & other_mask) == other_mask:
            count += 1
    return count


def greedy_init(seed):
    """Greedy 4-AP-free construction with random ordering."""
    rng = random.Random(seed)
    order = list(range(N))
    rng.shuffle(order)
    mask = 0
    for x in order:
        if is_4ap_free_add(mask, x):
            mask |= (1 << x)
    return mask


def local_search(mask, deadline):
    """Local search: add best, swap if stuck."""
    score = count_3aps_in_mask(mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try additions: find x not in mask that maximizes delta and is 4-AP-free
        best_add = -1
        best_add_delta = 0
        # Only consider x where is_4ap_free_add is true
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        for x in not_in_mask:
            if is_4ap_free_add(mask, x):
                d = delta_3aps_add(mask, x)
                if d > best_add_delta:
                    best_add_delta = d
                    best_add = x
        
        if best_add != -1:
            mask |= (1 << best_add)
            score += best_add_delta
            improved = True
            continue
        
        # Try swaps
        # For each x in mask (try removing), for each y not in mask (try adding)
        # New score = score - delta_remove(x) + delta_add(y, mask without x)
        # We need delta_add(y) computed on mask\x
        # To keep it tractable: iterate x in mask, then y not in mask
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in = [x for x in range(N) if not (mask >> x) & 1]
        
        best_swap_score = score
        best_swap_x = -1
        best_swap_y = -1
        
        for x in in_mask:
            dx = delta_3aps_remove(mask, x)
            new_mask = mask & ~(1 << x)
            for y in not_in:
                if y == x:
                    continue
                if not is_4ap_free_add(new_mask, y):
                    continue
                dy = delta_3aps_add(new_mask, y)
                new_score = score - dx + dy
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap_x = x
                    best_swap_y = y
        
        if best_swap_x != -1:
            mask = (mask & ~(1 << best_swap_x)) | (1 << best_swap_y)
            score = best_swap_score
            improved = True
            continue
        
        if not improved:
            break
    
    return mask, score


def mask_to_list(mask):
    return [i + 1 for i in range(N) if (mask >> i) & 1]


def genetic_algorithm(deadline):
    """Genetic algorithm with bitmask fitness."""
    pop_size = 40
    # Initialize population with greedy seeds + some random valid sets
    population = []
    for i in range(pop_size):
        population.append(greedy_init(i))
    
    # Score them
    scored = []
    for mask in population:
        s = count_3aps_in_mask(mask)
        scored.append((s, mask))
    scored.sort(key=lambda x: -x[0])
    
    best_score = scored[0][0]
    best_mask = scored[0][1]
    
    rng = random.Random(12345)
    generation = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Tournament selection + crossover + mutation
        new_pop = []
        # Keep top 5 as elites
        for s, m in scored[:5]:
            new_pop.append(m)
        
        while len(new_pop) < pop_size:
            # Select parents by tournament
            parent1 = scored[rng.randint(0, min(10, len(scored)-1))][1]
            parent2 = scored[rng.randint(0, min(10, len(scored)-1))][1]
            
            # Uniform crossover
            child = 0
            for i in range(N):
                if rng.random() < 0.5:
                    if (parent1 >> i) & 1:
                        child |= (1 << i)
                else:
                    if (parent2 >> i) & 1:
                        child |= (1 << i)
            
            # Mutation: flip some bits
            for i in range(N):
                if rng.random() < 0.05:
                    child ^= (1 << i)
            
            # Repair: remove any elements that create 4-APs
            # Build up from scratch: keep element only if it doesn't violate 4-AP
            repaired = 0
            # Order by something (e.g., index)
            for i in range(N):
                if (child >> i) & 1:
                    if is_4ap_free_add(repaired, i):
                        repaired |= (1 << i)
            
            # Optionally add more elements greedily to repair child
            for i in range(N):
                if not (repaired >> i) & 1:
                    if is_4ap_free_add(repaired, i):
                        repaired |= (1 << i)
            
            new_pop.append(repaired)
        
        # Evaluate new population
        scored = []
        for mask in new_pop:
            s = count_3aps_in_mask(mask)
            scored.append((s, mask))
        scored.sort(key=lambda x: -x[0])
        
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_mask = scored[0][1]
    
    # Polish the best
    final_mask, final_score = local_search(best_mask, deadline)
    if final_score > best_score:
        best_score = final_score
        best_mask = final_mask
    
    return best_mask, best_score


def hybrid_search(deadline):
    """Combine genetic exploration with local search exploitation."""
    start = time.time()
    # Phase 1: Genetic for ~60% of time
    ga_deadline = start + (deadline - start) * 0.6
    ga_mask, ga_score = genetic_algorithm(ga_deadline)
    
    # Phase 2: Local search on GA best
    final_mask, final_score = local_search(ga_mask, deadline)
    
    # Also try local search on a few greedy seeds
    for seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline - 2:
            break
        m = greedy_init(seed)
        lm, ls = local_search(m, deadline)
        if ls > final_score:
            final_score = ls
            final_mask = lm
    
    return final_mask, final_score


def construct():
    start = time.time()
    deadline = start + TIME_BUDGET
    
    # Try hybrid search
    mask, score = hybrid_search(deadline)
    
    return mask_to_list(mask)


# Fallback safety: ensure we return a valid nonempty set within time
def _construct_safe():
    start = time.time()
    deadline = start + TIME_BUDGET
    try:
        return construct()
    except Exception:
        # Fallback: simple greedy
        rng = random.Random(42)
        order = list(range(N))
        rng.shuffle(order)
        mask = 0
        for x in order:
            if is_4ap_free_add(mask, x):
                mask |= (1 << x)
        return mask_to_list(mask)

construct = _construct_safe

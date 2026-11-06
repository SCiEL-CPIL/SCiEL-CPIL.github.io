import time
import random

N = 512

# Precompute all 4-APs and their membership
_all_4aps = []
_ap_by_element = [[] for _ in range(N + 1)]

for a in range(1, N - 3):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        b = a + d
        c = a + 2 * d
        ap = (a, b, c, e)
        idx = len(_all_4aps)
        _all_4aps.append(ap)
        for elem in ap:
            _ap_by_element[elem].append(idx)

# For each element, precompute the other three elements in each AP it belongs to
_ap_partners = [None] * (N + 1)
for x in range(1, N + 1):
    partners = []
    for idx in _ap_by_element[x]:
        a, b, c, e = _all_4aps[idx]
        others = tuple(t for t in (a, b, c, e) if t != x)
        partners.append(others)
    _ap_partners[x] = tuple(partners)


def _cantor_bits():
    """Base-3 Cantor set: numbers with no digit 2 in base 3."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits


def _can_add(x, bits):
    """Check if x can be added to set represented by bits without creating a 4-AP."""
    for others in _ap_partners[x]:
        o1, o2, o3 = others
        if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
            return False
    return True


def _is_valid(bits):
    """Check if the set is 4-AP-free."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for others in _ap_partners[x]:
            o1, o2, o3 = others
            if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
                return False
    return True


def _bits_to_set(bits):
    return [i for i in range(1, N + 1) if (bits >> i) & 1]


def _greedy_forward(bits_start=0):
    """Forward greedy: add elements in order 1..N if possible."""
    bits = bits_start
    for x in range(1, N + 1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_reverse(bits_start=0):
    """Reverse greedy: add elements in order N..1 if possible."""
    bits = bits_start
    for x in range(N, 0, -1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random(bits_start=0, rng=None):
    """Random-order greedy."""
    if rng is None:
        rng = random.Random()
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = bits_start
    for x in order:
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_weighted(bits_start=0, rng=None):
    """Greedy with weighting: prefer elements with fewer conflicts."""
    if rng is None:
        rng = random.Random()
    bits = bits_start
    remaining = [x for x in range(1, N + 1) if not ((bits >> x) & 1)]
    
    while remaining:
        # Score each candidate by how many APs it would block
        scored = []
        for x in remaining:
            if _can_add(x, bits):
                # Count how many other elements this would block
                block_count = 0
                for others in _ap_partners[x]:
                    o1, o2, o3 = others
                    # Check if any of these are in remaining
                    if (bits >> o1) & 1 or (bits >> o2) & 1 or (bits >> o3) & 1:
                        block_count += 1
                scored.append((block_count, x))
        
        if not scored:
            break
        
        # Pick the one with minimum blocking (or random among minimum)
        min_block = min(s[0] for s in scored)
        candidates = [x for b, x in scored if b == min_block]
        x = rng.choice(candidates)
        
        bits |= (1 << x)
        remaining.remove(x)
    
    return bits


def _local_search_aggressive(bits, deadline, max_passes=50):
    """
    Aggressive local search: try to add elements, and if blocked, try swaps.
    Uses a more systematic approach for finding beneficial swaps.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # First try to add any element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # Try swaps: for each x not in set, find minimal removal set
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return current
                if (current >> x) & 1:
                    continue
                
                # Find all APs that would be completed
                blocking_aps = []
                for others in _ap_partners[x]:
                    o1, o2, o3 = others
                    if ((current >> o1) & 1) and ((current >> o2) & 1) and ((current >> o3) & 1):
                        blocking_aps.append((o1, o2, o3))
                
                if not blocking_aps:
                    continue
                
                # Collect all candidate removals
                candidates = set()
                for ap in blocking_aps:
                    for elem in ap:
                        candidates.add(elem)
                
                # Try each single removal
                found_swap = False
                for rem in sorted(candidates):
                    if time.time() > deadline - 2:
                        return current
                    trial = current & ~(1 << rem)
                    if _can_add(x, trial):
                        trial |= (1 << x)
                        # Try to re-add some elements
                        trial = _greedy_forward(trial)
                        trial_size = bin(trial).count('1')
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            found_swap = True
                            break
                
                # If no single swap worked, try 2-element removals for small cases
                if not found_swap and len(blocking_aps) <= 4 and time.time() < deadline - 3:
                    # Try pairs from the union of elements in blocking APs
                    all_elems = set()
                    for ap in blocking_aps:
                        all_elems.update(ap)
                    elem_list = list(all_elems)
                    if len(elem_list) <= 12:
                        for i in range(len(elem_list)):
                            if time.time() > deadline - 2:
                                return current
                            for j in range(i + 1, len(elem_list)):
                                trial = current & ~(1 << elem_list[i]) & ~(1 << elem_list[j])
                                if _can_add(x, trial):
                                    trial |= (1 << x)
                                    trial = _greedy_forward(trial)
                                    trial_size = bin(trial).count('1')
                                    if trial_size > current_size:
                                        current = trial
                                        current_size = trial_size
                                        improved = True
                                        found_swap = True
                                        break
                            if found_swap:
                                break
            
            if not improved:
                break
    
    return current


def _randomized_local_search(bits, deadline, rng):
    """
    Randomized version: shuffle the order in which we try to add elements,
    and randomize which removals to try first.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _pass in range(15):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # Random order for trying additions
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # Try a few random swaps
            for _attempt in range(80):
                if time.time() > deadline - 2:
                    return current
                x = rng.randint(1, N)
                if (current >> x) & 1:
                    continue
                
                blocking_aps = []
                for others in _ap_partners[x]:
                    o1, o2, o3 = others
                    if ((current >> o1) & 1) and ((current >> o2) & 1) and ((current >> o3) & 1):
                        blocking_aps.append((o1, o2, o3))
                
                if not blocking_aps:
                    continue
                
                candidates = set()
                for ap in blocking_aps:
                    candidates.update(ap)
                
                if not candidates:
                    continue
                
                rem = rng.choice(list(candidates))
                trial = current & ~(1 << rem)
                if _can_add(x, trial):
                    trial |= (1 << x)
                    trial = _greedy_forward(trial)
                    trial_size = bin(trial).count('1')
                    if trial_size > current_size:
                        current = trial
                        current_size = trial_size
                        improved = True
    
    return current


def _hill_climb(bits, deadline, rng):
    """
    Hill climbing: repeatedly try single-element swaps and accept if improvement.
    """
    current = bits
    current_size = bin(current).count('1')
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try adding any element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if improved:
            continue
        
        # Try single swaps
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current >> x) & 1:
                continue
            
            blocking_aps = []
            for others in _ap_partners[x]:
                o1, o2, o3 = others
                if ((current >> o1) & 1) and ((current >> o2) & 1) and ((current >> o3) & 1):
                    blocking_aps.append((o1, o2, o3))
            
            if not blocking_aps:
                continue
            
            candidates = set()
            for ap in blocking_aps:
                candidates.update(ap)
            
            for rem in candidates:
                if time.time() > deadline - 2:
                    break
                trial = current & ~(1 << rem)
                if _can_add(x, trial):
                    trial |= (1 << x)
                    trial = _greedy_forward(trial)
                    trial_size = bin(trial).count('1')
                    if trial_size > current_size:
                        current = trial
                        current_size = trial_size
                        improved = True
                        break
        
        if not improved:
            break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Multiple seed constructions
    cantor = _cantor_bits()
    fwd = _greedy_forward()
    rev = _greedy_reverse()
    
    seeds = [cantor, fwd, rev]
    seed_sizes = [bin(s).count('1') for s in seeds]
    
    best = seeds[seed_sizes.index(max(seed_sizes))]
    best_size = max(seed_sizes)
    
    # 2. Local search on best seed
    if time.time() < deadline - 30:
        improved = _local_search_aggressive(best, deadline - 10)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 3. Randomized restarts with different strategies
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 20 and restart_count < 40:
        restart_count += 1
        
        # Different strategies
        r = rng.random()
        if r < 0.2:
            # Start from Cantor, add randomly
            bits = _greedy_random(cantor, rng)
        elif r < 0.4:
            # Pure random greedy
            bits = _greedy_random(0, rng)
        elif r < 0.55:
            # Weighted greedy
            bits = _greedy_weighted(0, rng)
        elif r < 0.7:
            # Start from best, perturb: remove a random element, then re-greedy
            bits = best
            if best_size > 1:
                elems = _bits_to_set(bits)
                if elems:
                    rem = rng.choice(elems)
                    bits = bits & ~(1 << rem)
                    bits = _greedy_forward(bits)
        elif r < 0.85:
            # Reverse greedy from scratch
            bits = _greedy_reverse()
            # Then forward greedy on top
            bits = _greedy_forward(bits)
        else:
            # Start from a random subset, then greedy
            bits = 0
            for x in range(1, N + 1):
                if rng.random() < 0.3:
                    bits |= (1 << x)
            bits = _greedy_forward(bits)
        
        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size
        
        # Local search on this result
        if time.time() < deadline - 15:
            improved = _local_search_aggressive(bits, deadline - 10)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
    
    # 4. Final aggressive local search on best
    if time.time() < deadline - 5:
        improved = _local_search_aggressive(best, deadline)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 5. Hill climb polish
    if time.time() < deadline - 2:
        improved = _hill_climb(best, deadline, rng)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 6. Verify
    if not _is_valid(best):
        best = cantor
    
    return sorted(_bits_to_set(best))


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")

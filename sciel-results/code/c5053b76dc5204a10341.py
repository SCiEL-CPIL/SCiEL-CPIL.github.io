import time
import random

N = 256
DEADLINE = 80.0

# Precompute all 3-APs and 4-APs as lists of bitmasks
three_aps = []  # list of (a, b, c) with 0-indexed
for d in range(1, N // 2 + 1):
    for a in range(N - 2 * d):
        three_aps.append((a, a + d, a + 2 * d))

four_aps = []  # list of (a, b, c, e) with 0-indexed
for d in range(1, N // 3 + 1):
    for a in range(N - 3 * d):
        four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

# For each element, precompute which 3-APs and 4-APs it's in
elem_to_3aps = [[] for _ in range(N)]
for i, (a, b, c) in enumerate(three_aps):
    elem_to_3aps[a].append(i)
    elem_to_3aps[b].append(i)
    elem_to_3aps[c].append(i)

elem_to_4aps = [[] for _ in range(N)]
for i, (a, b, c, e) in enumerate(four_aps):
    elem_to_4aps[a].append(i)
    elem_to_4aps[b].append(i)
    elem_to_4aps[c].append(i)
    elem_to_4aps[e].append(i)


def count_3aps(mask):
    """Count distinct 3-term APs in bitmask."""
    count = 0
    for a, b, c in three_aps:
        if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
            count += 1
    return count


def count_4aps(mask):
    """Count nonconstant 4-term APs in bitmask."""
    count = 0
    for a, b, c, e in four_aps:
        if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1 and (mask >> e) & 1:
            count += 1
    return count


def has_4ap(mask):
    """Check if mask contains any 4-AP."""
    for a, b, c, e in four_aps:
        if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1 and (mask >> e) & 1:
            return True
    return False


def build_base_set():
    """Classic 4-AP-free set: numbers with no '1' in base-3 (0-indexed)."""
    mask = 0
    for x in range(N):
        n = x + 1  # convert to 1-indexed for base-3 check
        valid = True
        while n > 0:
            if n % 3 == 1:
                valid = False
                break
            n //= 3
        if valid:
            mask |= (1 << x)
    return mask


def greedy_construction(order=None):
    """Greedy: add elements in given order, skip if creates 4-AP."""
    if order is None:
        order = list(range(N))
    mask = 0
    for x in order:
        candidate = mask | (1 << x)
        if not has_4ap(candidate):
            mask = candidate
    return mask


def local_improvement(mask, time_budget=5.0):
    """Try additions and swaps to improve 3-AP count while maintaining 4-AP-free."""
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        current_score = count_3aps(mask)
        
        # Try adding each element
        for x in range(N):
            if (mask >> x) & 1:
                continue
            candidate = mask | (1 << x)
            if not has_4ap(candidate):
                new_score = count_3aps(candidate)
                if new_score > current_score:
                    mask = candidate
                    current_score = new_score
                    improved = True
        
        # Try swaps: remove one, add another
        if time.time() - start > time_budget:
            break
        for x_out in range(N):
            if not ((mask >> x_out) & 1):
                continue
            mask_without = mask & ~(1 << x_out)
            loss = current_score - count_3aps(mask_without)
            if loss <= 0:
                continue
            
            for x_in in range(N):
                if (mask >> x_in) & 1:
                    continue
                candidate = mask_without | (1 << x_in)
                if has_4ap(candidate):
                    continue
                new_score = count_3aps(candidate)
                if new_score > current_score:
                    mask = candidate
                    current_score = new_score
                    improved = True
                    break
    
    return mask


def simulated_annealing(mask, time_budget=20.0, seed=None):
    """SA: toggle bits, accept violations with penalty, optimize 3-AP count."""
    if seed is not None:
        random.seed(seed)
    
    best_mask = mask
    best_score = count_3aps(mask)
    current_mask = mask
    current_score = best_score
    temp = 15.0
    cooling = 0.9998
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if temp < 0.05:
            break
        
        x = random.randint(0, N - 1)
        new_mask = current_mask ^ (1 << x)
        
        four_count = count_4aps(new_mask)
        penalty = four_count * 100
        new_score = count_3aps(new_mask) - penalty
        
        delta = new_score - current_score
        if delta > 0 or random.random() < min(1.0, delta / (temp + 1e-9)):
            current_mask = new_mask
            current_score = new_score
            if four_count == 0 and new_score > best_score:
                best_mask = new_mask
                best_score = new_score
        
        temp *= cooling
        
        # Periodic repair if violations exist
        if four_count > 0 and random.random() < 0.05:
            # Remove elements to fix 4-APs
            fixed = current_mask
            changed = True
            while changed:
                changed = False
                for a, b, c, e in four_aps:
                    if (fixed >> a) & 1 and (fixed >> b) & 1 and (fixed >> c) & 1 and (fixed >> e) & 1:
                        # Remove the element that appears in most 4-APs in this set
                        freq = {a: 0, b: 0, c: 0, e: 0}
                        for aa, bb, cc, ee in four_aps:
                            if (fixed >> aa) & 1 and (fixed >> bb) & 1 and (fixed >> cc) & 1 and (fixed >> ee) & 1:
                                for el in (aa, bb, cc, ee):
                                    if el in freq:
                                        freq[el] += 1
                        worst = max(freq, key=lambda k: freq[k])
                        fixed &= ~(1 << worst)
                        changed = True
                        break
            
            rep_score = count_3aps(fixed)
            if rep_score > best_score:
                best_mask = fixed
                best_score = rep_score
                current_mask = fixed
                current_score = rep_score
    
    return best_mask, best_score


def construct():
    random.seed(42)
    start = time.time()
    
    # Phase 1: Generate multiple initial solutions
    initials = []
    
    # Base set (classic 4-AP-free)
    base = build_base_set()
    initials.append(base)
    
    # Greedy in natural order
    greedy_nat = greedy_construction()
    initials.append(greedy_nat)
    
    # Greedy in random orders
    for i in range(5):
        order = list(range(N))
        random.seed(i + 100)
        random.shuffle(order)
        initials.append(greedy_construction(order))
    
    # Local improvement on each initial
    best_mask = 0
    best_score = 0
    for init in initials:
        if time.time() - start > DEADLINE - 10:
            break
        improved = local_improvement(init, time_budget=2.0)
        score = count_3aps(improved)
        if score > best_score:
            best_score = score
            best_mask = improved
    
    # Phase 2: Simulated Annealing from best and other good starts
    elapsed = time.time() - start
    remaining = DEADLINE - elapsed - 3.0
    
    if remaining > 5.0:
        # SA from best
        sa_time = remaining * 0.6
        sa_mask, sa_score = simulated_annealing(best_mask, sa_time, seed=1)
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
        
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed - 2.0
        
        if remaining > 3.0:
            # SA from base set
            sa_mask2, sa_score2 = simulated_annealing(base, remaining * 0.5, seed=2)
            if sa_score2 > best_score:
                best_score = sa_score2
                best_mask = sa_mask2
    
    # Phase 3: Final local improvement
    elapsed = time.time() - start
    remaining = DEADLINE - elapsed - 1.0
    if remaining > 1.0:
        final_mask = local_improvement(best_mask, time_budget=remaining)
        final_score = count_3aps(final_mask)
        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask
    
    # Ensure validity
    if has_4ap(best_mask):
        # Repair by removing elements
        fixed = best_mask
        changed = True
        while changed:
            changed = False
            for a, b, c, e in four_aps:
                if (fixed >> a) & 1 and (fixed >> b) & 1 and (fixed >> c) & 1 and (fixed >> e) & 1:
                    fixed &= ~(1 << a)
                    changed = True
                    break
        best_mask = fixed
    
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result

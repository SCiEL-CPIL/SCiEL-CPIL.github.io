"""Combined: bitset scoring + multi-restart SA + difference-targeting greedy + incremental local search."""
import time
import random
import numpy as np

N = 32
MAX_VAL = 512
MAX_D = 255
TIME_LIMIT = 118  # safety margin before 120s limit


def score_bitset(bits):
    """Count distinct d>0 such that a, a+d, a+2d all in set, using bit operations."""
    count = 0
    for d in range(1, MAX_D + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count


def bits_to_list(bits):
    result = []
    for i in range(1, MAX_VAL + 1):
        if bits & (1 << i):
            result.append(i)
    return result


def list_to_bits(lst):
    bits = 0
    for x in lst:
        bits |= (1 << x)
    return bits


def get_covered_diffs(bits):
    """Return set of d values that are covered (have a 3-AP)."""
    covered = set()
    for d in range(1, MAX_D + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            covered.add(d)
    return covered


def diff_targeting_greedy(target_diffs, max_elements=N):
    """
    Greedy: place elements to cover target differences.
    target_diffs: list of d values to prioritize.
    Returns (bits, score).
    """
    bits = 0
    used = set()
    
    # Start with a small seed to anchor APs
    # Try to place elements that create APs for target diffs
    for d in target_diffs:
        if len(used) >= max_elements:
            break
        # For this d, we need a, a+d, a+2d all in set
        # Try to find a placement that maximizes new coverage
        best_gain = -1
        best_triple = None
        
        for a in range(1, MAX_VAL - 2 * d + 1):
            a2 = a + d
            a3 = a + 2 * d
            if a3 > MAX_VAL:
                break
            # Check how many of these are already in set
            in_count = sum(1 for x in (a, a2, a3) if x in used)
            # Gain = number of new elements that help cover d
            # Prefer triples where most elements are new and within range
            if in_count == 0:
                # All new - good if it creates a fresh AP
                gain = 1  # covers d
                # Bonus if elements also help other diffs
                candidate = (a, a2, a3)
            elif in_count == 1:
                # One element present, need 2 more
                gain = 1
                candidate = (a, a2, a3)
            else:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_triple = candidate
        
        if best_triple:
            for x in best_triple:
                if x not in used and x <= MAX_VAL:
                    used.add(x)
                    bits |= (1 << x)
    
    # Fill remaining slots with greedy marginal gain
    current_score = score_bitset(bits)
    for _ in range(max_elements - len(used)):
        best_gain = -1
        best_val = None
        for v in range(1, MAX_VAL + 1):
            if v in used:
                continue
            new_bits = bits | (1 << v)
            new_score = score_bitset(new_bits)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_val = v
        if best_val is None:
            break
        bits |= (1 << best_val)
        used.add(best_val)
        current_score = score_bitset(bits)
    
    return bits, current_score


def standard_greedy():
    """Standard greedy: add element with max marginal gain."""
    bits = 0
    current_score = 0
    used = set()
    
    for _ in range(N):
        best_gain = -1
        best_val = None
        for v in range(1, MAX_VAL + 1):
            if v in used:
                continue
            new_bits = bits | (1 << v)
            new_score = score_bitset(new_bits)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_val = v
        if best_val is None:
            break
        bits |= (1 << best_val)
        used.add(best_val)
        current_score = score_bitset(bits)
    
    return bits, current_score


def simulated_annealing(initial_bits, initial_score, time_budget, seed=None):
    """SA with swap moves on bitset representation."""
    if seed is not None:
        random.seed(seed)
    
    bits = initial_bits
    current_score = initial_score
    best_bits = bits
    best_score = current_score
    
    start_time = time.time()
    temp = 8.0
    cooling = 0.99997
    
    # Precompute list of positions for faster access
    def get_in_out(bits):
        in_set = []
        out_set = []
        for v in range(1, MAX_VAL + 1):
            if bits & (1 << v):
                in_set.append(v)
            else:
                out_set.append(v)
        return in_set, out_set
    
    while time.time() - start_time < time_budget:
        in_set, out_set = get_in_out(bits)
        if not in_set or not out_set:
            break
        
        # Swap: remove one from in_set, add one from out_set
        v_remove = random.choice(in_set)
        v_add = random.choice(out_set)
        new_bits = (bits & ~(1 << v_remove)) | (1 << v_add)
        
        new_score = score_bitset(new_bits)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
            bits = new_bits
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
        
        temp *= cooling
        if temp < 0.05:
            break
    
    return best_bits, best_score


def local_search_polish(bits, score, time_budget):
    """Exhaustive single-swap local search until no improvement."""
    best_bits = bits
    best_score = score
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        in_set = []
        out_set = []
        for v in range(1, MAX_VAL + 1):
            if best_bits & (1 << v):
                in_set.append(v)
            else:
                out_set.append(v)
        
        if not in_set or not out_set:
            break
        
        best_swap_gain = 0
        best_swap = None
        
        for v_remove in in_set:
            for v_add in out_set:
                new_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
                new_score = score_bitset(new_bits)
                gain = new_score - best_score
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (v_remove, v_add)
        
        if best_swap is not None and best_swap_gain > 0:
            v_remove, v_add = best_swap
            best_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
            best_score += best_swap_gain
        else:
            break
    
    return best_bits, best_score


def generate_diverse_initializations():
    """Generate diverse starting configurations."""
    candidates = []
    
    # 1. Consecutive range
    candidates.append(list(range(1, 33)))
    
    # 2. Sparse AP
    candidates.append(list(range(1, 513, 16)))  # 32 elements
    
    # 3. Small differences targeted
    small_diffs = list(range(1, 16))
    bits, _ = diff_targeting_greedy(small_diffs)
    candidates.append(bits_to_list(bits))
    
    # 4. Mixed small and large differences
    mixed_diffs = list(range(1, 10)) + list(range(20, 40, 3))
    bits, _ = diff_targeting_greedy(mixed_diffs)
    candidates.append(bits_to_list(bits))
    
    # 5. Random sets
    for _ in range(3):
        candidates.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # 6. Structured: cluster + spread
    cluster = random.sample(range(1, 100), 16)
    spread = random.sample(range(300, 513), 16)
    candidates.append(cluster + spread)
    
    # 7. Evenly spread
    geo = []
    for i in range(N):
        val = int(1 + i * (511 / 31.0))
        val = min(max(val, 1), MAX_VAL)
        geo.append(val)
    seen = set()
    unique_geo = []
    for v in geo:
        while v in seen and v < MAX_VAL:
            v += 1
        if v <= MAX_VAL:
            seen.add(v)
            unique_geo.append(v)
    while len(unique_geo) < N:
        v = random.randint(1, MAX_VAL)
        if v not in seen:
            seen.add(v)
            unique_geo.append(v)
    candidates.append(unique_geo)
    
    return candidates


def construct():
    start_time = time.time()
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    best_bits = None
    best_score = 0
    
    # Phase 1: Generate diverse initializations and polish each briefly
    candidates = generate_diverse_initializations()
    
    # Also run standard greedy
    greedy_bits, greedy_score = standard_greedy()
    if greedy_score > best_score:
        best_score = greedy_score
        best_bits = greedy_bits
    
    # Polish each candidate briefly
    for cand in candidates:
        if time_left() < 15:
            break
        bits = list_to_bits(cand)
        score = score_bitset(bits)
        
        # Brief local search
        ls_bits, ls_score = local_search_polish(bits, score, min(3, time_left() / 10))
        if ls_score > best_score:
            best_score = ls_score
            best_bits = ls_bits
    
    if time_left() < 5:
        return bits_to_list(best_bits)
    
    # Phase 2: Multiple SA rounds from best, with occasional restarts
    remaining = time_left()
    num_rounds = 4
    sa_budget = remaining * 0.5 / num_rounds
    polish_budget = remaining * 0.3 / num_rounds
    restart_budget = remaining * 0.2 / num_rounds
    
    for r in range(num_rounds):
        if time_left() < 5:
            break
        
        # SA from current best
        sa_bits, sa_score = simulated_annealing(best_bits, best_score, sa_budget, seed=r * 1000)
        if sa_score > best_score:
            best_score = sa_score
            best_bits = sa_bits
        
        # Local search polish
        ls_bits, ls_score = local_search_polish(best_bits, best_score, polish_budget)
        if ls_score > best_score:
            best_score = ls_score
            best_bits = ls_bits
        
        # Occasional restart with diff-targeting
        if r % 2 == 1 and time_left() > 10:
            target_diffs = list(range(1, 20))
            rt_bits, rt_score = diff_targeting_greedy(target_diffs)
            rt_bits, rt_score = local_search_polish(rt_bits, rt_score, min(restart_budget, time_left() / 2))
            if rt_score > best_score:
                best_score = rt_score
                best_bits = rt_bits
    
    # Final polish
    if time_left() > 2:
        final_bits, final_score = local_search_polish(best_bits, best_score, max(0, time_left() - 1))
        if final_score > best_score:
            best_bits = final_bits
            best_score = final_score
    
    return bits_to_list(best_bits)

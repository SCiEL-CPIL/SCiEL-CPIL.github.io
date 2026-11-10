"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _greedy_build(order, counts, chosen, deadline):
    """Add elements from order in sequence, skipping those that would violate B2[2]."""
    for x in order:
        if time.time() > deadline:
            break
        # Check if adding x would cause any sum to exceed count 2
        # New sums: x+x, and x+y for all y in chosen where y <= x (since we add x to existing set)
        # Actually, when we add x, the new pairs are (x, x) and (x, y) for all y in chosen
        # But we need to consider a <= b. If y > x, then the pair is (x, y) with x < y.
        # If y <= x, the pair is (y, x) with y <= x.
        # Since chosen is a set of distinct integers, for each y in chosen:
        #   if y <= x: pair (y, x), sum = x + y
        #   if y > x: pair (x, y), sum = x + y
        # So in all cases, the sum is x + y for each y in chosen, plus the pair (x, x).
        # We need to check that for each such sum, the current count + 1 <= 2,
        # i.e., current count <= 1.
        
        # Check x + x
        s = 2 * x
        if counts.get(s, 0) >= 2:
            continue
        
        # Check x + y for all y in chosen
        ok = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                ok = False
                break
        if not ok:
            continue
        
        # Add x
        chosen.append(x)
        counts[s] = counts.get(s, 0) + 1
        for y in chosen:
            if y < x:  # y was already in chosen before we appended x
                s = x + y
                counts[s] = counts.get(s, 0) + 1
    
    return chosen, counts

def _build_counts(chosen):
    """Build the sum counts dictionary for a given set."""
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            s = chosen[i] + chosen[j]
            counts[s] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    N = 1000
    
    # First, do a simple greedy in increasing order as a baseline
    chosen = []
    counts = defaultdict(int)
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        # Check if adding x is valid
        s = 2 * x
        if counts.get(s, 0) >= 2:
            continue
        ok = True
        for y in chosen:
            s2 = x + y
            if counts.get(s2, 0) >= 2:
                ok = False
                break
        if not ok:
            continue
        # Add x
        chosen.append(x)
        counts[s] = counts.get(s, 0) + 1
        for y in chosen:
            if y < x:
                s2 = x + y
                counts[s2] = counts.get(s2, 0) + 1
    
    best_size = len(chosen)
    best_result = list(chosen)
    
    if time.time() > deadline:
        return best_result
    
    # Now try multiple random strategies
    # Strategy: randomize the order, with bias toward either low or high numbers
    strategies = []
    
    # Strategy 1: Pure random
    strategies.append(('random', list(range(1, N + 1))))
    random.shuffle(strategies[0][1])
    
    # Strategy 2: Descending
    strategies.append(('desc', list(range(N, 0, -1))))
    
    # Strategy 3: Random with bias toward high
    strat3 = list(range(1, N + 1))
    random.shuffle(strat3)
    # Sort by random key but with a bias: multiply by a random factor
    # Actually, let's just create a random permutation with high numbers more likely to be early
    # Simple approach: take random order, but swap to put larger numbers earlier with some probability
    # Let's do: for each position, pick from remaining with probability proportional to value
    remaining = set(range(1, N + 1))
    strat3 = []
    for _ in range(N):
        if not remaining:
            break
        # Pick randomly from remaining, but weighted toward larger
        # Simple: pick from top half with 70% probability
        sorted_rem = sorted(remaining)
        mid = len(sorted_rem) // 2
        if random.random() < 0.7:
            # Pick from upper half
            candidates = sorted_rem[mid:]
            x = random.choice(candidates)
        else:
            x = random.choice(sorted_rem)
        strat3.append(x)
        remaining.remove(x)
    strategies.append(('biased_high', strat3))
    
    # Strategy 4: Random with bias toward low
    remaining = set(range(1, N + 1))
    strat4 = []
    for _ in range(N):
        if not remaining:
            break
        sorted_rem = sorted(remaining)
        mid = len(sorted_rem) // 2
        if random.random() < 0.7:
            candidates = sorted_rem[:mid + 1]
            x = random.choice(candidates)
        else:
            x = random.choice(sorted_rem)
        strat4.append(x)
        remaining.remove(x)
    strategies.append(('biased_low', strat4))
    
    # Strategy 5: Interleaved - alternate between low and high
    low_half = list(range(1, N // 2 + 1))
    high_half = list(range(N // 2 + 1, N + 1))
    random.shuffle(low_half)
    random.shuffle(high_half)
    strat5 = []
    i = j = 0
    while i < len(low_half) or j < len(high_half):
        if j < len(high_half) and (i >= len(low_half) or random.random() < 0.5):
            strat5.append(high_half[j])
            j += 1
        else:
            strat5.append(low_half[i])
            i += 1
    strategies.append(('interleaved', strat5))
    
    # Run each strategy with a time budget
    time_per_strategy = (deadline - time.time()) / max(1, len(strategies))
    
    for name, order in strategies:
        if time.time() > deadline - 5:  # Reserve 5s
            break
        
        str_deadline = min(deadline - 5, time.time() + time_per_strategy)
        
        chosen_try = []
        counts_try = defaultdict(int)
        
        for x in order:
            if time.time() > str_deadline:
                break
            s = 2 * x
            if counts_try.get(s, 0) >= 2:
                continue
            ok = True
            for y in chosen_try:
                s2 = x + y
                if counts_try.get(s2, 0) >= 2:
                    ok = False
                    break
            if not ok:
                continue
            chosen_try.append(x)
            counts_try[s] = counts_try.get(s, 0) + 1
            for y in chosen_try:
                if y < x:
                    s2 = x + y
                    counts_try[s2] = counts_try.get(s2, 0) + 1
        
        if len(chosen_try) > best_size:
            best_size = len(chosen_try)
            best_result = list(chosen_try)
    
    # Local improvement: try to add elements that were skipped
    if time.time() < deadline - 10 and best_result:
        counts_best = _build_counts(best_result)
        used = set(best_result)
        improvement = True
        while improvement and time.time() < deadline - 10:
            improvement = False
            # Try to add each unused number
            for x in range(1, N + 1):
                if x in used:
                    continue
                # Check if we can add x
                s = 2 * x
                if counts_best.get(s, 0) >= 2:
                    continue
                ok = True
                for y in best_result:
                    s2 = x + y
                    if counts_best.get(s2, 0) >= 2:
                        ok = False
                        break
                if not ok:
                    continue
                # Add x
                best_result.append(x)
                used.add(x)
                counts_best[s] = counts_best.get(s, 0) + 1
                for y in best_result:
                    if y < x:
                        s2 = x + y
                        counts_best[s2] = counts_best.get(s2, 0) + 1
                improvement = True
                best_size = len(best_result)
    
    return best_result
# EVOLVE-BLOCK-END

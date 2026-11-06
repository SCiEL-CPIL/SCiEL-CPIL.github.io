"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _is_sidon(S):
    """Check if S is a Sidon set: all unordered pair sums (i<=j) are unique."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = S[i] + S[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _max_sidon_subset_size(s):
    """Exhaustively find the largest Sidon subset of s."""
    n = len(s)
    s_sorted = sorted(s)
    best = 0
    for size in range(n, best, -1):
        for combo in combinations(range(n), size):
            subset = [s_sorted[i] for i in combo]
            if _is_sidon(subset):
                return size
    return best

def _generate_collision_candidates():
    """Generate candidate sets with heavy additive collision structure."""
    candidates = []
    rng = random.Random(42)
    
    # Type A: 6 pairs all summing to the same total
    for total in range(20, 130):
        a_min = max(0, total - 128)
        a_max = min(63, total // 2 - 1)
        if a_max >= a_min:
            available = list(range(a_min, a_max + 1))
            if len(available) >= 6:
                for _ in range(200):
                    a_vals = rng.sample(available, 6)
                    b_vals = [total - a for a in a_vals]
                    s = a_vals + b_vals
                    if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
                        candidates.append(s)
    
    # Type B: Two groups of 6, each with internal collision structure
    for _ in range(500):
        # Group 1: 3 pairs summing to t1
        t1 = rng.randint(10, 130)
        a_min1 = max(0, t1 - 128)
        a_max1 = min(63, t1 // 2 - 1)
        if a_max1 >= a_min1:
            avail1 = list(range(a_min1, a_max1 + 1))
            if len(avail1) >= 3:
                a1 = rng.sample(avail1, 3)
                b1 = [t1 - a for a in a1]
                g1 = a1 + b1
                # Group 2: 3 pairs summing to t2, different from g1
                t2 = rng.randint(10, 130)
                a_min2 = max(0, t2 - 128)
                a_max2 = min(63, t2 // 2 - 1)
                if a_max2 >= a_min2:
                    avail2 = list(range(a_min2, a_max2 + 1))
                    if len(avail2) >= 3:
                        a2 = rng.sample(avail2, 3)
                        b2 = [t2 - a for a in a2]
                        g2 = a2 + b2
                        s = g1 + g2
                        if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
                            candidates.append(s)
    
    # Type C: Arithmetic progression clusters
    for _ in range(300):
        # Two APs of length 6
        start1 = rng.randint(0, 50)
        d1 = rng.randint(1, 5)
        g1 = [start1 + i * d1 for i in range(6)]
        start2 = rng.randint(50, 120)
        d2 = rng.randint(1, 5)
        g2 = [start2 + i * d2 for i in range(6)]
        s = g1 + g2
        if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
            candidates.append(s)
    
    # Type D: Mix of pairs with different totals but overlapping sums
    for _ in range(300):
        s = []
        for i in range(6):
            a = rng.randint(0, 60)
            b = a + rng.randint(1, 60)
            if b <= 128:
                s.extend([a, b])
        if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
            candidates.append(s)
    
    return candidates

def _hill_climb(start_set, best_score, deadline, rng):
    """Local search: mutate elements, accept if score improves."""
    current = start_set[:]
    current_score = _max_sidon_subset_size(current)
    best = current[:]
    best_s = current_score
    
    while time.time() < deadline:
        if current_score >= best_s:
            current = best[:]
            current_score = best_s
        
        # Mutate 1-3 elements
        num_mut = rng.randint(1, 3)
        candidate = current[:]
        for _ in range(num_mut):
            idx = rng.randint(0, 11)
            new_val = rng.randint(0, 128)
            attempts = 0
            while new_val in candidate and attempts < 50:
                new_val = rng.randint(0, 128)
                attempts += 1
            if new_val not in candidate:
                candidate[idx] = new_val
            else:
                continue
        
        if len(set(candidate)) != 12:
            continue
        
        score = _max_sidon_subset_size(candidate)
        if score < best_s:
            best = candidate[:]
            best_s = score
            current = candidate[:]
            current_score = score
            if best_s <= 5:
                break
        
        current = candidate[:]
        current_score = score
    
    return best, best_s

def construct():
    """Search for an ambient set of 12 distinct integers in [0,128] that minimizes
    the largest Sidon subset size."""
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    # Baseline: pairs summing to 22
    baseline = [0, 22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17]
    best_set = baseline
    best_score = _max_sidon_subset_size(baseline)
    
    # Generate structural candidates
    candidates = _generate_collision_candidates()
    
    # Evaluate structural candidates and keep the best few for hill-climbing
    evaluated = []
    for s in candidates:
        if time.time() > deadline - 20:
            break
        score = _max_sidon_subset_size(s)
        evaluated.append((score, s))
        if score < best_score:
            best_score = score
            best_set = s[:]
    
    # Sort by score and do hill-climbing on the best ones
    evaluated.sort(key=lambda x: x[0])
    
    # Hill-climb on top candidates
    for score, s in evaluated[:20]:
        if time.time() > deadline - 5:
            break
        if score >= best_score:
            continue
        # Only hill-climb if this candidate is already good
        h_set, h_score = _hill_climb(s, best_score, deadline, rng)
        if h_score < best_score:
            best_score = h_score
            best_set = h_set
    
    # Also hill-climb from the best structural candidate even if not better than baseline
    if time.time() < deadline - 5:
        h_set, h_score = _hill_climb(best_set, best_score, deadline, rng)
        if h_score < best_score:
            best_score = h_score
            best_set = h_set
    
    # Final: try hill-climbing from baseline if time remains
    if time.time() < deadline - 5 and best_score > 5:
        h_set, h_score = _hill_climb(baseline, best_score, deadline, rng)
        if h_score < best_score:
            best_score = h_score
            best_set = h_set
    
    return best_set
# EVOLVE-BLOCK-END

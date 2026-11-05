import time
import random
from itertools import combinations

def _can_add(s, x):
    """
    Check if x can be added to valid set s without violating constraints.
    Two checks needed:
    1. x does not divide sum of any 3 distinct elements of s
    2. No element a in s divides (x + b + c) for any distinct b, c in s, b != a, c != a
    """
    if not s:
        return True
    s_list = sorted(s)
    n = len(s_list)
    s_set = set(s_list)
    
    # Check 1: x vs triples from s
    if n >= 3:
        max3 = s_list[-1] + s_list[-2] + s_list[-3]
        if x <= max3:
            min3 = s_list[0] + s_list[1] + s_list[2]
            lo = (min3 + x - 1) // x
            hi = max3 // x
            if lo <= hi:
                for k in range(lo, hi + 1):
                    target = k * x
                    # 3-sum check in sorted list
                    found = False
                    for i in range(n - 2):
                        a1 = s_list[i]
                        if a1 + s_list[i+1] + s_list[i+2] > target:
                            break
                        for j in range(i + 1, n - 1):
                            a2 = s_list[j]
                            if a1 + a2 + s_list[j+1] > target:
                                break
                            rem = target - a1 - a2
                            if rem > a2 and rem in s_set:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    
    # Check 2: each a in s vs triples containing x
    for idx_a, a in enumerate(s_list):
        # Triples: x + b + c where b, c in s, b != a, c != a, b != c
        others = [s_list[i] for i in range(n) if i != idx_a]
        m = len(others)
        if m < 2:
            continue
        max_sum = x + others[-1] + others[-2]
        if a > max_sum:
            continue
        min_sum = x + others[0] + others[1]
        lo = (min_sum + a - 1) // a
        hi = max_sum // a
        if lo > hi:
            continue
        others_set = set(others)
        for k in range(lo, hi + 1):
            target = k * a
            need = target - x
            if need < 0:
                continue
            # 2-sum check
            found = False
            for i in range(m - 1):
                b = others[i]
                if b + others[i+1] > need:
                    break
                rem = need - b
                if rem > b and rem in others_set:
                    found = True
                    break
            if found:
                return False
    return True


def _is_valid(s):
    """Full validity check for a set s."""
    s_list = sorted(s)
    n = len(s_list)
    if n < 4:
        return True
    s_set = set(s_list)
    for idx_a, a in enumerate(s_list):
        others = [s_list[i] for i in range(n) if i != idx_a]
        m = len(others)
        if m < 3:
            continue
        max3 = others[-1] + others[-2] + others[-3]
        if a > max3:
            continue
        min3 = others[0] + others[1] + others[2]
        lo = (min3 + a - 1) // a
        hi = max3 // a
        if lo > hi:
            continue
        for k in range(lo, hi + 1):
            target = k * a
            found = False
            for i in range(m - 2):
                a1 = others[i]
                if a1 + others[i+1] + others[i+2] > target:
                    break
                for j in range(i + 1, m - 1):
                    a2 = others[j]
                    if a1 + a2 + others[j+1] > target:
                        break
                    rem = target - a1 - a2
                    if rem > a2 and rem in s_set:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True


def _greedy_descending():
    """Greedy: add numbers from 192 down to 1 if valid."""
    s = set()
    for x in range(192, 0, -1):
        if _can_add(s, x):
            s.add(x)
    return s


def _greedy_primes_first():
    """Add large primes first, then composites."""
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    s = set()
    for x in range(192, 0, -1):
        if is_prime(x) and _can_add(s, x):
            s.add(x)
    for x in range(192, 0, -1):
        if x not in s and _can_add(s, x):
            s.add(x)
    return s


def _local_search(s, rng, deadline, max_rounds=300):
    """
    Local search: try to add elements, or swap (remove one, add one), or 2-swap.
    """
    s = set(s)
    cur_size = len(s)
    all_nums = set(range(1, 193))
    stall = 0
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding (prefer larger)
        candidates = sorted(all_nums - s, reverse=True)
        rng.shuffle(candidates)
        for x in candidates[:20]:
            if _can_add(s, x):
                s.add(x)
                cur_size += 1
                improved = True
                stall = 0
                break
        if improved:
            continue
        
        # Try 1-swap: remove small, add large
        elems = sorted(s)
        # Try removing from smallest
        for rem in elems[:10]:
            candidates = sorted(all_nums - s, reverse=True)
            rng.shuffle(candidates)
            for add in candidates[:15]:
                if add == rem:
                    continue
                new_s = (s - {rem}) | {add}
                if _is_valid(new_s):
                    s = new_s
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try 2-swap
        if rng.random() < 0.5:
            elems = sorted(s)
            rem_pairs = list(combinations(elems[:12], 2))
            rng.shuffle(rem_pairs)
            candidates = sorted(all_nums - s, reverse=True)
            add_pairs = list(combinations(candidates[:12], 2))
            rng.shuffle(add_pairs)
            for rem1, rem2 in rem_pairs[:8]:
                for add1, add2 in add_pairs[:8]:
                    new_s = (s - {rem1, rem2}) | {add1, add2}
                    if _is_valid(new_s):
                        s = new_s
                        improved = True
                        stall = 0
                        break
                if improved:
                    break
        if not improved:
            stall += 1
            if stall > 40:
                break
    
    return s


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline (90s limit, 10s safety)
    rng = random.Random(12345)
    
    # Initial solutions from different greedy strategies
    best = _greedy_descending()
    best_size = len(best)
    
    alt = _greedy_primes_first()
    if len(alt) > best_size:
        best = alt
        best_size = len(alt)
    
    # Local search from best
    result = _local_search(best, rng, deadline, max_rounds=500)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Try more rounds with different starting points
    for rnd in range(50):
        if time.time() > deadline:
            break
        
        # Diverse start: random valid subset biased large
        k = rng.randint(30, 45)
        weights = [x * x for x in range(1, 193)]  # bias toward large
        start_s = set(rng.choices(range(1, 193), weights=weights, k=k))
        # Make it valid
        valid_s = set()
        for x in sorted(start_s, reverse=True):
            if _can_add(valid_s, x):
                valid_s.add(x)
        
        res = _local_search(valid_s, rng, deadline, max_rounds=200)
        if len(res) > best_size:
            best = res
            best_size = len(res)
        
        # Also try local search from current best with perturbation
        if best_size >= 40:
            perturbed = set(best)
            # Remove 2-3 random small elements
            small = sorted(perturbed)[:5]
            for x in rng.sample(small, min(2, len(small))):
                perturbed.discard(x)
            res2 = _local_search(perturbed, rng, deadline, max_rounds=200)
            if len(res2) > best_size:
                best = res2
                best_size = len(res2)
    
    # Final: try to add any remaining
    all_nums = set(range(1, 193))
    for x in sorted(all_nums - best, reverse=True):
        if _can_add(best, x):
            best.add(x)
    
    return list(best)

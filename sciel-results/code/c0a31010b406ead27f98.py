# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def is_valid_fast(s):
    """Faster validity check: for each element a, check if any 3 distinct others sum to multiple of a."""
    lst = sorted(s)
    n = len(lst)
    if n < 4:
        return True
    # For each element a, check if any 3 distinct others sum to a multiple of a
    # Key insight: only need to check multiples of a that are achievable (3*min to 3*max)
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        # Find achievable range of 3-sums from others
        others_sorted = sorted(others)
        min_sum3 = others_sorted[0] + others_sorted[1] + others_sorted[2]
        max_sum3 = others_sorted[-1] + others_sorted[-2] + others_sorted[-3]
        
        # Check all triples of others
        for i in range(m):
            for j in range(i+1, m):
                s2 = others[i] + others[j]
                for k in range(j+1, m):
                    if (s2 + others[k]) % a == 0:
                        return False
    return True

def greedy_construct(seed_rng=None):
    """Greedy from largest to smallest."""
    rng = seed_rng or random
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        # Check validity
        valid = True
        lst = sorted(candidate)
        n = len(lst)
        for idx_a, a in enumerate(lst):
            others = lst[:idx_a] + lst[idx_a+1:]
            m = len(others)
            if m < 3:
                continue
            for i in range(m):
                for j in range(i+1, m):
                    s2 = others[i] + others[j]
                    for k in range(j+1, m):
                        if (s2 + others[k]) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    rng = random.Random(42)
    
    # Start with greedy
    best = greedy_construct()
    best_set = frozenset(best)
    best_size = len(best)
    
    all_numbers = set(range(1, 193))
    
    def local_search(s, rng, max_iters=200):
        """Hill-climb: try adding elements, removing and replacing."""
        s = set(s)
        cur_size = len(s)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding a random element not in s
            candidates = list(all_numbers - s)
            rng.shuffle(candidates)
            for x in candidates[:8]:  # try up to 8 random additions
                new_s = s | {x}
                if is_valid_fast(new_s):
                    s = new_s
                    cur_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one, add another
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:4]:
                candidates = list(all_numbers - s)
                rng.shuffle(candidates)
                for add in candidates[:8]:
                    new_s = (s - {rem}) | {add}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return s
    
    # Run multiple local search rounds with diverse starting points
    for round_num in range(300):
        if time.time() > deadline:
            break
        
        # Generate diverse starting points
        if round_num == 0:
            start_s = set(best)
        elif round_num == 1:
            start_s = greedy_construct()
        else:
            # Random start: take a random subset and filter to valid
            k = rng.randint(35, 45)
            start_s = set(rng.sample(range(1, 193), k))
            valid_s = set()
            for x in sorted(start_s, reverse=True):
                new_s = valid_s | {x}
                if is_valid_fast(new_s):
                    valid_s = new_s
            start_s = valid_s
        
        # Local search
        result = local_search(start_s, rng, max_iters=150)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = frozenset(result)
            best = list(result)
    
    return list(best_set)
# EVOLVE-BLOCK-END

import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check if vals satisfy the cardinality-separated subset sum property.
        Returns True if valid. Uses bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        max_sum = sum(vals)
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        sums = [0] * (n + 1)
        sums[0] = 1  # sum 0 with 0 elements
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check pairwise disjointness of sums[k] for different k
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    def try_greedy_descending(pool, max_size):
        """Try to build a valid set by greedily adding elements from largest to smallest."""
        result = []
        for v in sorted(pool, reverse=True):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_greedy_ascending(pool, max_size):
        """Try to build a valid set by greedily adding elements from smallest to largest."""
        result = []
        for v in sorted(pool):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_grow(initial, all_vals):
        """Grow a valid initial set by trying to add remaining values (smallest first)."""
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        return cur

    def try_randomized(pool, max_size, iterations, rng):
        """Try randomized orderings to find larger valid sets."""
        best = []
        for _ in range(iterations):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= max_size:
                    break
                candidate = result + [v]
                if check_valid(candidate):
                    result = candidate
            if len(result) > len(best):
                best = result
                if len(best) >= max_size:
                    break
        return best

    pool = list(range(1, N + 1))
    best = []
    
    # Strategy 1: Superincreasing baseline
    sup = [1]
    for i in range(1, 60):
        nxt = sum(sup) + 1
        if nxt <= N:
            sup.append(nxt)
        else:
            break
    if check_valid(sup):
        best = list(sup)
    
    # Strategy 2: Descending greedy
    desc = try_greedy_descending(pool, MAX_SIZE)
    if len(desc) > len(best):
        best = desc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 3: Ascending greedy
    asc = try_greedy_ascending(pool, MAX_SIZE)
    if len(asc) > len(best):
        best = asc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 4: Grow from superincreasing
    grown = try_grow(sup, pool)
    if len(grown) > len(best):
        best = grown
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 5: Try subsets of the pool - maybe not all values are useful
    for start in range(1, 50):
        if time.time() > deadline - 5:
            break
        sub_pool = list(range(start, N + 1))
        desc = try_greedy_descending(sub_pool, MAX_SIZE)
        if len(desc) > len(best):
            best = desc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 6: Randomized attempts
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(42)
        # Estimate how many iterations we can afford
        iterations = min(100, int(remaining_time * 30))
        rand_best = try_randomized(pool, MAX_SIZE, iterations, rng)
        if len(rand_best) > len(best):
            best = rand_best
    
    # Strategy 7: Random small seeds then grow
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(12345)
        for trial in range(min(50, int(remaining_time * 10))):
            if time.time() > deadline - 1:
                break
            size = rng.randint(1, 6)
            seed_vals = rng.sample(range(1, N + 1), size)
            if not check_valid(seed_vals):
                continue
            cur = try_grow(sorted(seed_vals), pool)
            if len(cur) > len(best):
                best = cur
    
    # Ensure we return a list of Python ints
    best = [int(x) for x in best]
    return best

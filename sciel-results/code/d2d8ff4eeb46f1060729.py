"""Rewritten program: randomized local search with fast divisibility-chain validity."""
import random
import time

# EVOLVE-BLOCK-START

def _subset_sums(nums):
    """Return list of all nonempty subset sums (with duplicates)."""
    sums = [0]
    for x in nums:
        sums += [s + x for s in sums]
    # drop the initial 0
    return sums[1:]

def _is_valid(nums):
    """Check that no two distinct nonempty subset sums have one dividing the other.
    Equal sums from different subsets are also invalid (handled implicitly:
    if s appears twice, then s divides s, but we need distinct subsets ->
    actually equal sums s,s: s divides s, and they come from distinct subsets,
    so invalid. Our check: for each sum value v, if count>=2 then invalid because
    v divides v and there are two distinct subsets. Also for v and k*v with k>=2
    from distinct subsets, invalid.
    """
    from collections import Counter
    sums = _subset_sums(nums)
    cnt = Counter(sums)
    # any value appearing >=2 times -> invalid (two distinct subsets same sum)
    for v, c in cnt.items():
        if c >= 2:
            return False
    # now each sum value is unique; check divisibility among distinct values
    vals = sorted(cnt.keys())
    n = len(vals)
    # for each pair where a < b, check if b % a == 0
    # optimize: for each a, check multiples
    for i in range(n):
        a = vals[i]
        if a <= 1:
            # 1 divides everything; if 1 is a subset sum and there's any other sum, invalid
            if n > 1:
                return False
            continue
        for j in range(i + 1, n):
            b = vals[j]
            if b % a == 0:
                return False
    return True

def _greedy_build(rng, max_size=30, pool=None, deadline=None):
    """Greedy: start empty, add elements that keep validity, prefer smaller to leave room."""
    if pool is None:
        pool = list(range(1, 257))
    chosen = []
    available = list(pool)
    while len(chosen) < max_size:
        if deadline is not None and time.time() > deadline:
            break
        best = None
        # try adding each candidate; pick one that keeps validity
        # heuristic: try smaller numbers first to allow more elements
        for x in sorted(available):
            cand = chosen + [x]
            if _is_valid(cand):
                best = x
                break
        if best is None:
            break
        chosen.append(best)
        available.remove(best)
    return chosen

def _local_search(start, rng, deadline, max_iter=20000):
    """Hill-climb: try swapping/adding elements to increase size while valid."""
    cur = list(start)
    best = list(cur)
    cur_size = len(cur)
    pool = [x for x in range(1, 257) if x not in cur]
    t0 = time.time()
    stuck = 0
    while len(cur) < 30:
        if time.time() > deadline:
            break
        # try to add a random element
        added = False
        for _ in range(20):
            if time.time() > deadline:
                break
            x = rng.choice(pool)
            cand = cur + [x]
            if _is_valid(cand):
                cur = cand
                pool.remove(x)
                added = True
                if len(cur) > len(best):
                    best = list(cur)
                break
        if not added:
            # try a swap: remove one, add another
            if not pool:
                break
            removed = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = rng.randrange(len(cur))
                x = rng.choice(pool)
                cand = cur[:idx] + cur[idx+1:] + [x]
                if _is_valid(cand):
                    cur = cand
                    pool.remove(x)
                    pool.append(cur[idx])
                    removed = True
                    if len(cur) > len(best):
                        best = list(cur)
                    break
            if not removed:
                stuck += 1
                if stuck > 50:
                    break
        else:
            stuck = 0
    return best

def construct():
    deadline = time.time() + 75  # safety margin under 90s
    rng = random.Random(42)
    best = []
    # Multiple restarts with different strategies
    for seed in range(50):
        if time.time() > deadline:
            break
        r = random.Random(seed)
        # Strategy: greedy with shuffled pool
        pool = list(range(1, 257))
        r.shuffle(pool)
        cand = _greedy_build(r, max_size=30, pool=pool, deadline=deadline)
        if len(cand) > len(best):
            best = cand
        # Local search from greedy result
        if len(cand) >= 2:
            improved = _local_search(cand, r, deadline, max_iter=5000)
            if len(improved) > len(best):
                best = improved
        # Also try a pure random hill-climb start
        if time.time() < deadline:
            random_start = []
            avail = list(range(1, 257))
            r.shuffle(avail)
            for x in avail:
                if len(random_start) >= 30:
                    break
                if _is_valid(random_start + [x]):
                    random_start.append(x)
            if len(random_start) > len(best):
                best = random_start
            improved = _local_search(random_start, r, deadline, max_iter=5000)
            if len(improved) > len(best):
                best = improved
        if len(best) >= 30:
            break
    # Final polish: try to add more
    if len(best) < 30:
        final = _local_search(best, rng, deadline, max_iter=10000)
        if len(final) > len(best):
            best = final
    # Ensure Python ints and list
    result = [int(x) for x in best]
    # Verify distinct and in range
    assert len(set(result)) == len(result), "duplicates"
    assert all(1 <= x <= 256 for x in result), "out of range"
    return result

# EVOLVE-BLOCK-END

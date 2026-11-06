"""Improved program: greedy baselines + bounded DFS with deadline guard."""
import time
import random

DEADLINE = 85.0  # safety margin under 90s kill

def _validate(seq):
    """Return True if strictly increasing, in [2,128], and all block products distinct."""
    n = len(seq)
    if n == 0:
        return True
    if any(not (2 <= x <= 128) for x in seq):
        return False
    for i in range(n - 1):
        if seq[i + 1] <= seq[i]:
            return False
    seen = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in seen:
                return False
            seen.add(p)
    return True

def _greedy(order):
    """Greedy: iterate candidates in given order, add if no tail-product collision."""
    values, products = [], set()
    for x in order:
        # compute new tail products (blocks ending at x)
        tails = [x]
        p = x
        for a in reversed(values):
            p *= a
            tails.append(p)
        if not (products & set(tails)):
            values.append(x)
            products.update(tails)
    return values

def _dfs_max(deadline):
    """Depth-first search maximizing length; returns best valid sequence."""
    best = []
    start = time.time()

    # candidates: we can only use each integer at most once, and sequence must be strictly increasing.
    # We choose an increasing subsequence of [2..128].
    allnums = list(range(2, 129))

    def search(values, products, idx):
        nonlocal best
        if time.time() > deadline:
            return
        if len(values) > len(best):
            best = list(values)
        # try extending with a number greater than values[-1]
        last = values[-1] if values else 1
        for x in allnums:
            if x <= last:
                continue
            if time.time() > deadline:
                return
            # new tails: blocks ending at x
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            # collision check
            hit = False
            for t in new_tails:
                if t in products:
                    hit = True
                    break
            if hit:
                continue
            # also need new_tails internally distinct? They are all multiples, strictly increasing since factors >=2, so distinct.
            values.append(x)
            products.update(new_tails)
            search(values, products, x)
            values.pop()
            products.difference_update(new_tails)
            if time.time() > deadline:
                return

    search([], set(), 1)
    return best

def _local_search(seq, deadline):
    """Try to improve sequence by swapping or inserting elements."""
    if not seq:
        return seq
    best = list(seq)
    start = time.time()
    
    # Try inserting each unused number at each position
    allnums = set(range(2, 129))
    used = set(best)
    unused = sorted(allnums - used)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(best) + 1):
            if time.time() > deadline:
                break
            for x in unused:
                # Check if we can insert x at position pos
                # Must maintain strictly increasing
                if pos > 0 and best[pos - 1] >= x:
                    continue
                if pos < len(best) and best[pos] <= x:
                    continue
                # Try inserting
                candidate = best[:pos] + [x] + best[pos:]
                if _validate(candidate):
                    best = candidate
                    used.add(x)
                    unused.remove(x)
                    improved = True
                    break
            if improved:
                break
    
    return best

def _beam_search(width, deadline):
    """Beam search to find long valid sequences."""
    start = time.time()
    allnums = list(range(2, 129))
    
    # Each beam entry: (sequence, products_set)
    beams = [([], set())]
    
    while time.time() < deadline:
        new_beams = []
        for seq, products in beams:
            last = seq[-1] if seq else 1
            for x in allnums:
                if x <= last:
                    continue
                # Compute new tails
                new_tails = [x]
                p = x
                for a in reversed(seq):
                    p *= a
                    new_tails.append(p)
                # Check collision
                if products & set(new_tails):
                    continue
                new_seq = seq + [x]
                new_products = products | set(new_tails)
                new_beams.append((new_seq, new_products))
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if not new_beams:
            break
        
        # Keep top 'width' beams by length
        new_beams.sort(key=lambda b: len(b[0]), reverse=True)
        beams = new_beams[:width]
        
        # If all beams have same length and we can't extend, break
        if len(beams) == 0:
            break
        max_len = len(beams[0][0])
        if all(len(b[0]) == max_len for b in beams):
            # Check if any can be extended
            can_extend = False
            for seq, products in beams:
                last = seq[-1] if seq else 1
                for x in allnums:
                    if x <= last:
                        continue
                    new_tails = [x]
                    p = x
                    for a in reversed(seq):
                        p *= a
                        new_tails.append(p)
                    if not (products & set(new_tails)):
                        can_extend = True
                        break
                if can_extend:
                    break
            if not can_extend:
                break
    
    if not beams:
        return []
    return beams[0][0]

def construct():
    start = time.time()
    deadline = start + DEADLINE

    results = []

    # 1) Greedy baselines with various orderings
    base = list(range(2, 129))
    results.append(_greedy(base))                       # ascending
    results.append(_greedy(list(reversed(base))))       # descending
    # primes first
    primes = [x for x in base if all(x % d for d in range(2, int(x ** 0.5) + 1))]
    comps = [x for x in base if x not in set(primes)]
    results.append(_greedy(primes + comps))
    results.append(_greedy(comps + primes))

    def _t(x):
        # count prime factors with multiplicity
        n, c = x, 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                n //= d
                c += 1
            d += 1
        if n > 1:
            c += 1
        return c
    results.append(_greedy(sorted(base, key=_t)))
    results.append(_greedy(sorted(base, key=lambda x: -_t(x))))

    # Add some more diverse orderings
    # Sort by number of divisors
    def _num_divisors(x):
        count = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
    results.append(_greedy(sorted(base, key=_num_divisors)))
    results.append(_greedy(sorted(base, key=lambda x: -_num_divisors(x))))

    # keep best so far
    best = max(results, key=len) if results else []
    if not _validate(best):
        best = []

    # 2) Beam search to try to beat the best greedy length
    if time.time() < deadline - 5.0:
        try:
            beam_res = _beam_search(width=50, deadline=deadline - 2.0)
            if beam_res and len(beam_res) > len(best):
                if _validate(beam_res):
                    best = beam_res
        except Exception:
            pass

    # 3) Bounded DFS to try to beat the best found length
    if time.time() < deadline - 2.0:
        try:
            dfs_res = _dfs_max(deadline - 1.0)
            if dfs_res and len(dfs_res) > len(best):
                if _validate(dfs_res):
                    best = dfs_res
        except Exception:
            pass

    # 4) Local search polish
    if best and time.time() < deadline - 1.0:
        try:
            polished = _local_search(best, deadline - 0.5)
            if _validate(polished) and len(polished) > len(best):
                best = polished
        except Exception:
            pass

    # final safety: ensure valid; if not, fall back to a simple valid greedy
    if not best or not _validate(best):
        fb = _greedy(base)
        best = fb if _validate(fb) else list(range(2, 129, 2))

    if time.time() > start + DEADLINE:
        # extremely unlikely; return best valid anyway
        pass

    return best

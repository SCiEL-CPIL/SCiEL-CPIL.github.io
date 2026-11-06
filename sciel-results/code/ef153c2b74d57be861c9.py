import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0

    N = 256
    ALL = (1 << N) - 1

    # Precompute conflict masks using the algebraic characterization:
    # a,b conflict iff (a/gcd(a,b) + b/gcd(a,b)) | gcd(a,b)
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        conflict_mask[i] = m

    def popcount(x):
        return x.bit_count()

    def greedy_complete(mask, deadline, max_rounds=50):
        """Repeatedly scan all candidates and add any that fit."""
        cur = mask
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            changed = False
            not_in = ALL & ~cur
            while not_in:
                if time.time() > deadline:
                    break
                lsb = not_in & (-not_in)
                e = lsb.bit_length() - 1
                not_in ^= lsb
                if (conflict_mask[e] & cur) == 0:
                    cur |= lsb
                    changed = True
            if not changed:
                break
        return cur

    def local_search(start_mask, deadline, max_iters=200000, seed=12345):
        best_mask = start_mask
        best_size = popcount(start_mask)
        cur = start_mask
        cur_size = best_size
        rng = random.Random(seed)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            r = rng.random()
            if r < 0.5:
                # ADD a random compatible element
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                if (conflict_mask[e] & cur) == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            elif r < 0.85:
                # 1-SWAP: remove a conflicting element of e, add e
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                conflicts = conflict_mask[e] & cur
                if conflicts == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
                    continue
                # pick a random conflicting element to remove
                k = rng.randrange(popcount(conflicts))
                c = None
                cc = conflicts
                for _ in range(k + 1):
                    lsb = cc & (-cc)
                    c = lsb.bit_length() - 1
                    cc ^= lsb
                new_mask = (cur & ~(1 << c)) | (1 << e)
                if (conflict_mask[e] & (new_mask & ~(1 << e))) == 0:
                    cur = new_mask
                    cur = greedy_complete(cur, deadline, max_rounds=3)
                    cur_size = popcount(cur)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            else:
                # 2-REMOVE + complete
                drop = 1 if rng.random() < 0.5 else 2
                trial = cur
                for _ in range(drop):
                    if trial == 0:
                        break
                    e = rng.randrange(N)
                    if (trial >> e) & 1:
                        trial &= ~(1 << e)
                trial = greedy_complete(trial, deadline, max_rounds=5)
                ts = popcount(trial)
                if ts > cur_size:
                    cur = trial
                    cur_size = ts
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur

        return best_mask

    # Base: all odds (mutually non-conflicting since coprime odds don't conflict,
    # and for non-coprime odds the condition is restrictive)
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)

    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    best_overall = base_mask
    best_size = popcount(base_mask)

    def consider(mask):
        nonlocal best_overall, best_size
        sz = popcount(mask)
        if sz > best_size:
            best_overall = mask
            best_size = sz

    # Multi-start greedy with diverse orderings
    # Strategy: start with odds, try adding evens in various orders
    orderings = []
    
    # Natural order
    orderings.append(evens_indices[:])
    # Reverse
    orderings.append(list(reversed(evens_indices)))
    # By conflict degree with base
    deg = [(popcount(conflict_mask[e] & base_mask), e) for e in evens_indices]
    orderings.append([e for _, e in sorted(deg)])
    orderings.append([e for _, e in sorted(deg, key=lambda x: -x[0])])
    
    # Random orderings
    for _ in range(30):
        order = evens_indices[:]
        random.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > DEADLINE - 30:
            break
        m = greedy_complete(base_mask | 0, DEADLINE - 30)  # just complete base first
        # Actually start from base and add evens in order
        cur = base_mask
        for e in order:
            if (cur >> e) & 1:
                continue
            if (conflict_mask[e] & cur) == 0:
                cur |= (1 << e)
        cur = greedy_complete(cur, DEADLINE - 30, max_rounds=10)
        consider(cur)

    # Local search from best
    if time.time() < DEADLINE - 20:
        improved = local_search(best_overall, DEADLINE - 20, max_iters=100000, seed=42)
        consider(improved)

    # More local search with different seeds
    if time.time() < DEADLINE - 10:
        improved = local_search(best_overall, DEADLINE - 10, max_iters=80000, seed=123)
        consider(improved)

    # Final polish: try all single additions
    if time.time() < DEADLINE - 3:
        m = best_overall
        for e in range(N):
            if (m >> e) & 1:
                continue
            if (conflict_mask[e] & m) == 0:
                m |= (1 << e)
        consider(m)

    # Final local search
    if time.time() < DEADLINE - 1:
        improved = local_search(best_overall, DEADLINE - 1, max_iters=30000, seed=999)
        consider(improved)

    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])

    # Final verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                return sorted([i + 1 for i in odds_indices])

    return result

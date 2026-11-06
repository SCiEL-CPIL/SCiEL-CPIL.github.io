import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill

    N = 256
    FULL = (1 << N) - 1

    # Precompute conflict masks: cm[i] has bit j set if (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    cm = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        cm[i] = m

    odds = [i for i in range(N) if (i + 1) % 2 == 1]
    evens = [i for i in range(N) if (i + 1) % 2 == 0]

    def popcount(x):
        return x.bit_count()

    def is_valid(m):
        for i in range(N):
            if (m >> i) & 1:
                if cm[i] & m:
                    return False
        return True

    def greedy_add(start, order):
        m = start
        for e in order:
            if (m >> e) & 1:
                continue
            if cm[e] & m:
                continue
            m |= (1 << e)
        return m

    def local_search(m, iters):
        best = m
        bestsz = popcount(m)
        for it in range(iters):
            if time.time() > DEADLINE - 2:
                break
            r = random.random()
            if r < 0.5:
                # Try to add a random element
                e = random.randrange(N)
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
            elif r < 0.85:
                # 1-swap: remove a conflicted element, add another
                e = random.randrange(N)
                if (m >> e) & 1:
                    continue
                conf = cm[e] & m
                if not conf:
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                else:
                    # Pick the conflicted element with fewest other conflicts
                    best_c = -1
                    best_cconf = N + 1
                    tmp = conf
                    while tmp:
                        lb = tmp & -tmp
                        c = lb.bit_length() - 1
                        tmp ^= lb
                        cconf = popcount(cm[c] & (m & ~(1 << c)))
                        if cconf < best_cconf:
                            best_cconf = cconf
                            best_c = c
                    if best_c >= 0:
                        newm = (m & ~(1 << best_c)) | (1 << e)
                        if not (cm[e] & (newm & ~(1 << e))):
                            m = newm
                            s = popcount(m)
                            if s > bestsz:
                                best, bestsz = m, s
            else:
                # 2-swap: remove two, add two
                if popcount(m) < 2:
                    continue
                e1 = random.randrange(N)
                e2 = random.randrange(N)
                if (m >> e1) & 1 or (m >> e2) & 1 or e1 == e2:
                    continue
                conf1 = cm[e1] & m
                conf2 = cm[e2] & m
                if not conf1 and not conf2:
                    m |= (1 << e1) | (1 << e2)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                    continue
                # Find good removal candidates
                rem_candidates = []
                tmp = (conf1 | conf2)
                while tmp:
                    lb = tmp & -tmp
                    c = lb.bit_length() - 1
                    tmp ^= lb
                    cconf = popcount(cm[c] & (m & ~(1 << c)))
                    rem_candidates.append((cconf, c))
                rem_candidates.sort()
                # Try removing top 2 and adding e1, e2
                if len(rem_candidates) >= 2:
                    c1 = rem_candidates[0][1]
                    c2 = rem_candidates[1][1]
                    if c1 == c2:
                        continue
                    newm = (m & ~(1 << c1) & ~(1 << c2)) | (1 << e1) | (1 << e2)
                    # Verify validity
                    valid = True
                    for i in range(N):
                        if (newm >> i) & 1:
                            if cm[i] & newm:
                                valid = False
                                break
                    if valid:
                        m = newm
                        s = popcount(m)
                        if s > bestsz:
                            best, bestsz = m, s
        return best

    # Simulated annealing
    def simulated_annealing(m, iters, t_start, t_end):
        best = m
        bestsz = popcount(m)
        cur = m
        cursz = popcount(cur)
        for it in range(iters):
            if time.time() > DEADLINE - 2:
                break
            temp = t_start + (t_end - t_start) * (it / max(iters, 1))
            r = random.random()
            if r < 0.4:
                # Add
                e = random.randrange(N)
                if (cur >> e) & 1:
                    continue
                if not (cm[e] & cur):
                    newm = cur | (1 << e)
                    newsz = cursz + 1
                else:
                    continue
            elif r < 0.8:
                # 1-swap
                e = random.randrange(N)
                if (cur >> e) & 1:
                    continue
                conf = cm[e] & cur
                if not conf:
                    newm = cur | (1 << e)
                    newsz = cursz + 1
                else:
                    tmp = conf
                    best_c = -1
                    best_cconf = N + 1
                    while tmp:
                        lb = tmp & -tmp
                        c = lb.bit_length() - 1
                        tmp ^= lb
                        cconf = popcount(cm[c] & (cur & ~(1 << c)))
                        if cconf < best_cconf:
                            best_cconf = cconf
                            best_c = c
                    if best_c < 0:
                        continue
                    newm = (cur & ~(1 << best_c)) | (1 << e)
                    newsz = cursz
            else:
                # Remove
                if cursz == 0:
                    continue
                tmp = cur
                lb = tmp & -tmp
                c = lb.bit_length() - 1
                newm = cur & ~(1 << c)
                newsz = cursz - 1

            delta = newsz - cursz
            if delta >= 0 or random.random() < pow(2.718281828, delta / max(temp, 1e-10)):
                cur = newm
                cursz = newsz
                if cursz > bestsz:
                    best, bestsz = cur, cursz
        return best

    # Branch and bound for final polish
    sys.setrecursionlimit(10000)

    def branch_bound(candidates, current_size, current_mask, depth):
        nonlocal best, bestsz
        if time.time() > DEADLINE - 1:
            return
        if not candidates:
            if current_size > bestsz:
                bestsz = current_size
                best = current_mask
            return

        ub = current_size + popcount(candidates)
        if ub <= bestsz:
            return

        if depth > 30:
            return

        # Pick vertex with max degree in candidates
        c = candidates
        best_v = -1
        max_deg = -1
        while c:
            lb = c & -c
            v = lb.bit_length() - 1
            c ^= lb
            deg = popcount(candidates & cm[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v

        if best_v < 0:
            return

        # Branch 1: include best_v
        new_cand = candidates & ~(1 << best_v) & ~cm[best_v]
        branch_bound(new_cand, current_size + 1, current_mask | (1 << best_v), depth + 1)

        # Branch 2: exclude best_v
        new_cand = candidates & ~(1 << best_v)
        branch_bound(new_cand, current_size, current_mask, depth + 1)

    # Initialize
    best = 0
    bestsz = 0

    def consider(m):
        nonlocal best, bestsz
        if not is_valid(m):
            return
        s = popcount(m)
        if s > bestsz:
            best, bestsz = m, s

    # Phase 1: Greedy constructions
    base = 0
    for i in odds:
        base |= (1 << i)
    consider(base)

    # Greedy evens on odds with various orders
    consider(greedy_add(base, evens))
    consider(greedy_add(base, list(reversed(evens))))

    for _ in range(20):
        if time.time() > DEADLINE - 30:
            break
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(base, ro))

    # Random cores
    for _ in range(20):
        if time.time() > DEADLINE - 30:
            break
        core = 0
        for i in random.sample(range(N), random.randint(1, 8)):
            if not (cm[i] & core):
                core |= (1 << i)
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(core, ro))

    # Phase 2: Local search from best
    if time.time() < DEADLINE - 20:
        best = local_search(best, 100000)

    # Phase 3: Simulated annealing
    if time.time() < DEADLINE - 15:
        best = simulated_annealing(best, 200000, 50.0, 0.1)

    # Phase 4: More local search
    if time.time() < DEADLINE - 10:
        best = local_search(best, 80000)

    # Phase 5: Another SA pass
    if time.time() < DEADLINE - 8:
        best = simulated_annealing(best, 150000, 30.0, 0.05)

    # Phase 6: Final local search
    if time.time() < DEADLINE - 5:
        best = local_search(best, 50000)

    # Phase 7: Branch and bound polish on a restricted candidate set
    if time.time() < DEADLINE - 3:
        # Find elements not in best that could potentially be added
        not_in_best = FULL & ~best
        # Restrict to elements that don't conflict with too many in best
        limited_cand = 0
        for i in range(N):
            if (not_in_best >> i) & 1:
                if popcount(cm[i] & best) <= 2:
                    limited_cand |= (1 << i)
        # Also include some elements from best that might be swappable
        if popcount(limited_cand) > 0:
            # Try BB on a small subset
            sample_size = min(15, popcount(limited_cand))
            cand_list = []
            tmp = limited_cand
            while tmp and len(cand_list) < sample_size:
                lb = tmp & -tmp
                c = lb.bit_length() - 1
                tmp ^= lb
                cand_list.append(c)
            # Run BB starting from best with these candidates
            # Actually, let's just try adding them greedily with BB
            for c in cand_list:
                if not (cm[c] & best):
                    best |= (1 << c)
                    bestsz = popcount(best)

    # Final validation
    if not is_valid(best):
        # Fallback to all odds
        best = 0
        for i in odds:
            best |= (1 << i)

    if best == 0:
        best = 1  # number 1 alone

    result = sorted([i + 1 for i in range(N) if (best >> i) & 1])
    return result

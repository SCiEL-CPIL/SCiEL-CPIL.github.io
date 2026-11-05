import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback (general, simple)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Pre-compute neighbors and bitsets
    neighbors = [None] * total_points
    masks = [0] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        ball = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            nd = (orig_digit + 1) % q
            delta = -2 * power if nd == 0 else power
            ball.append(idx + delta)
            nd = (orig_digit - 1) % q
            delta = 2 * power if nd == 2 else -power
            ball.append(idx + delta)
        neighbors[idx] = ball
        m = 0
        for p in ball:
            m |= (1 << p)
        masks[idx] = m

    # reverse neighbors: for each point, which balls contain it
    rev_neighbors = [[] for _ in range(total_points)]
    for b in range(total_points):
        for p in neighbors[b]:
            rev_neighbors[p].append(b)

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    # ---------- Greedy construction ----------
    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        while True:
            if not np.any(uncovered_counts > 0):
                break
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            best_idx = int(rng.choice(candidates))
            selected.append(best_idx)
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors[p]:
                        uncovered_counts[b_prime] -= 1
        return selected

    # ---------- Pruning (remove redundant codewords) ----------
    def prune(selected_indices):
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.min(coverage_count[neighbors[idx]]) > 1:
                    selected_list.pop(i)
                    coverage_count[neighbors[idx]] -= 1
                    changed = True
                    break
        return selected_list

    # ---------- Local search with targeted swaps ----------
    def local_search(selected_indices, deadline):
        selected_list = prune(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        risky_mask = 0
        for i in range(total_points):
            if coverage_count[i] == 1:
                risky_mask |= (1 << i)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = np.random.default_rng(999)
        stall = 0
        max_stall = 20000

        while time.time() < deadline and stall < max_stall:
            removed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if (masks[idx] & risky_mask) == 0:
                    selected_list.pop(i)
                    for p in neighbors[idx]:
                        coverage_count[p] -= 1
                        if coverage_count[p] == 1:
                            risky_mask |= (1 << p)
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                    removed = True
                    stall = 0
                    break
            if removed:
                continue
            if not selected_list:
                break

            c = selected_list[int(rng.integers(len(selected_list)))]
            c_mask = masks[c]
            # Targeted: candidate codewords covering all risky points covered by c
            risky_in_c = c_mask & risky_mask
            if risky_in_c == 0:
                # c covers no risky point; we can't remove, but try neighbor swap
                candidates = set(neighbors[c][1:])
            else:
                # intersect neighbor lists of risky points in c
                rp = [p for p in neighbors[c] if (risky_mask >> p) & 1]
                if rp:
                    common = None
                    for p in rp:
                        s = set(rev_neighbors[p])
                        common = s if common is None else (common & s)
                    if common is None:
                        common = set()
                    candidates = set(common)
                else:
                    candidates = set()
            # add neighbor-restricted + randoms
            for p in neighbors[c][1:]:
                candidates.add(p)
            for _ in range(6):
                candidates.add(int(rng.integers(total_points)))
            candidates.discard(c)

            swapped = False
            for c_prime in candidates:
                if c_prime in selected_list:
                    continue
                if (c_mask & ~masks[c_prime] & risky_mask) == 0:
                    pos = selected_list.index(c)
                    selected_list[pos] = c_prime
                    for p in neighbors[c]:
                        coverage_count[p] -= 1
                        if coverage_count[p] == 1:
                            risky_mask |= (1 << p)
                    for p in neighbors[c_prime]:
                        coverage_count[p] += 1
                        if coverage_count[p] == 2:
                            risky_mask &= ~(1 << p)
                    swapped = True
                    stall = 0
                    break
            if not swapped:
                stall += 1
        return best_code

    # ---------- Genetic algorithm ----------
    def is_valid(code):
        cov = 0
        for idx in code:
            cov |= masks[idx]
        full = (1 << total_points) - 1
        return cov == full

    def crossover(p1, p2):
        rng = np.random.default_rng()
        # exchange disjoint spatial partitions (positions) between parents
        perm = list(range(n))
        rng.shuffle(perm)
        # random split into two groups
        k = int(rng.integers(1, n))
        group_a = set(perm[:k])
        # word for a codeword: take positions in group_a from p1, rest from p2
        # need to map to index
        def make_word(w1, w2):
            w = [0] * n
            for pos in range(n):
                w[pos] = w1[pos] if pos in group_a else w2[pos]
            return w
        # build words
        def idx2w(i):
            w = [0] * n
            t = i
            for pos in range(n):
                w[pos] = t % q
                t //= q
            return w
        def w2idx(w):
            idx = 0
            for pos in range(n):
                idx += w[pos] * powers[pos]
            return idx
        w1 = [idx2w(i) for i in p1]
        w2 = [idx2w(i) for i in p2]
        child_words = []
        for a, b in zip(w1, w2):
            child_words.append(make_word(a, b))
        # also swap halves to create diversity
        child_words2 = [make_word(b, a) for a, b in zip(w1, w2)]
        child1 = [w2idx(w) for w in child_words]
        child2 = [w2idx(w) for w in child_words2]
        return child1, child2

    def mutate(code, deadline):
        # removal + repair
        code = list(code)
        rng = np.random.default_rng()
        # try to remove a redundant codeword
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            coverage_count[neighbors[idx]] += 1
        for _ in range(3):
            i = int(rng.integers(len(code)))
            idx = code[i]
            if np.min(coverage_count[neighbors[idx]]) > 1:
                code.pop(i)
                coverage_count[neighbors[idx]] -= 1
        # repair: add codewords to cover any gaps (shouldn't be gaps after removal)
        # verify coverage
        cov = 0
        for idx in code:
            cov |= masks[idx]
        full = (1 << total_points) - 1
        if cov != full:
            # repair with greedy on uncovered
            uncovered = full & ~cov
            # find uncovered points
            pts = [p for p in range(total_points) if (uncovered >> p) & 1]
            for p in pts:
                # pick a ball covering p with max new coverage
                best = None
                best_gain = -1
                for b in rev_neighbors[p]:
                    if b in code:
                        continue
                    gain = bin(masks[b] & uncovered).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best = b
                if best is not None:
                    code.append(best)
                    cov |= masks[best]
                    uncovered = full & ~cov
                    if uncovered == 0:
                        break
        return code

    def genetic(population, deadline):
        rng = np.random.default_rng(42)
        full = (1 << total_points) - 1
        pop = population[:]
        # evaluate
        for i in range(len(pop)):
            if not is_valid(pop[i]):
                # repair
                pop[i] = mutate(pop[i], deadline)
        best_code = min(pop, key=len)
        best_len = len(best_code)
        gen = 0
        while time.time() < deadline:
            gen += 1
            if gen % 20 == 0:
                break
            # sort by length
            pop.sort(key=len)
            # elitism: keep top
            elite = pop[:max(1, len(pop)//3)]
            new_pop = list(elite)
            # produce offspring
            while len(new_pop) < len(pop) and time.time() < deadline:
                # select two parents (tournament)
                def select():
                    a = pop[int(rng.integers(len(pop)))]
                    b = pop[int(rng.integers(len(pop)))]
                    return a if len(a) <= len(b) else b
                p1 = select()
                p2 = select()
                c1, c2 = crossover(p1, p2)
                # mutate
                if rng.random() < 0.7:
                    c1 = mutate(c1, deadline)
                if rng.random() < 0.7:
                    c2 = mutate(c2, deadline)
                # ensure valid
                if is_valid(c1):
                    new_pop.append(c1)
                if is_valid(c2) and len(new_pop) < len(pop):
                    new_pop.append(c2)
            # add some random greedy to maintain diversity
            if time.time() < deadline:
                g = fast_greedy(int(rng.integers(1000)))
                g = prune(g)
                new_pop.append(g)
            pop = new_pop[:len(pop)]
            # track best
            for code in pop:
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    best_code = None
    best_len = float('inf')

    # Build initial population via greedy with different seeds
    population = []
    for seed in range(6):
        if time.time() > deadline - 40:
            break
        sel = fast_greedy(seed)
        sel = prune(sel)
        population.append(sel)
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel[:]

    if best_code is None:
        best_code = prune(fast_greedy(0))
        best_len = len(best_code)

    # Local search on best
    best_code = local_search(best_code, deadline)
    best_code = prune(best_code)
    best_len = len(best_code)

    # Genetic algorithm
    if time.time() < deadline - 10:
        ga_code = genetic(population, deadline)
        ga_code = prune(ga_code)
        if len(ga_code) < best_len:
            best_len = len(ga_code)
            best_code = ga_code

    # Final local search + prune
    if time.time() < deadline - 5:
        best_code = local_search(best_code, deadline)
    best_code = prune(best_code)

    result = [idx_to_word(idx) for idx in best_code]
    return result

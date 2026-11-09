import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    random.seed(98765)

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def compute_ap_contributions(lst):
        """For each element lst[i], compute which d values have lst[i] as the
        middle term of an AP (a, lst[i], a+2d) where a=lst[i]-d, a+2d=lst[i]+d.
        Also count total score."""
        s_set = set(lst)
        # For each d, find all middle elements
        # middle[i] = count of a such that a-d and a+d are in set (a = lst[i])
        # We need: for d in 1..255, lst[i] is middle iff (lst[i]-d in s_set) and (lst[i]+d in s_set)
        middle_counts = [0] * N  # number of d where lst[i] is middle
        # Actually we need per-d tracking for delta
        # Let's just compute the full score and track which d are satisfied
        satisfied_d = set()
        for d in range(1, 256):
            for i in range(N):
                x = lst[i]
                if (x - d) in s_set and (x + d) in s_set:
                    satisfied_d.add(d)
                    break
        return len(satisfied_d)

    def delta_score(lst, mask, old_val, new_val):
        """Compute score change when replacing old_val with new_val in lst/mask."""
        new_mask = mask ^ (1 << old_val) ^ (1 << new_val)
        # For each d, check if AP status changes
        delta = 0
        for d in range(1, 256):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta

    def gen_initial_population(pop_size):
        pop = []
        seen = set()

        # Interval
        lst = list(range(1, N + 1))
        m = list_to_mask(lst)
        pop.append((lst[:], m, score_mask(m)))
        seen.add(tuple(sorted(lst)))

        # APs with various steps
        for step in range(5, 16):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                m = list_to_mask(lst)
                pop.append((lst[:], m, score_mask(m)))
                seen.add(tuple(sorted(lst)))

        # Two-scale structures: A + k*B
        for mult in [2, 3, 4, 5, 7]:
            for trial in range(20):
                sizeA = random.randint(4, 9)
                sizeB = random.randint(4, 9)
                A = set(random.sample(range(1, 40), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, min(maxB, 80) + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    candidates = sorted(S)
                    # Try multiple samples
                    for _ in range(5):
                        lst = random.sample(candidates, N)
                        m = list_to_mask(lst)
                        s = score_mask(m)
                        key = tuple(sorted(lst))
                        if key not in seen:
                            seen.add(key)
                            pop.append((lst[:], m, s))

        # Random
        while len(pop) < pop_size:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                pop.append((lst[:], m, s))
        return pop

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(pop)
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def crossover(p1, p2):
        """Blend crossover: take sorted elements from both parents, keep top N by value."""
        s1 = set(p1[0])
        s2 = set(p2[0])
        # Union, then select N elements preferring those in both
        common = s1 & s2
        only1 = s1 - s2
        only2 = s2 - s1
        # Build candidate list
        cands = list(common)
        # Fill remaining from only1 and only2
        remaining = N - len(cands)
        extra = list(only1) + list(only2)
        random.shuffle(extra)
        cands.extend(extra[:remaining])
        # If still short, add random
        cset = set(cands)
        while len(cands) < N:
            v = random.randint(1, MAX_VAL)
            if v not in cset:
                cset.add(v)
                cands.append(v)
        cands = cands[:N]
        random.shuffle(cands)
        m = list_to_mask(cands)
        s = score_mask(m)
        return (cands, m, s)

    def mutate(individual, rate=0.12):
        lst = individual[0][:]
        s_set = set(lst)
        m = individual[1]
        s = individual[2]
        for i in range(N):
            if random.random() < rate:
                old = lst[i]
                new = random.randint(1, MAX_VAL)
                attempts = 0
                while new in s_set and attempts < 30:
                    new = random.randint(1, MAX_VAL)
                    attempts += 1
                if new not in s_set:
                    d = delta_score(lst, m, old, new)
                    if d > 0 or random.random() < 0.15:  # accept some negative moves
                        s_set.discard(old)
                        s_set.add(new)
                        lst[i] = new
                        m ^= (1 << old) ^ (1 << new)
                        s += d
        return (lst, m, s)

    def local_search(individual, max_iter=80):
        """Hill climbing with random and targeted replacements."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try each position
            for i in range(N):
                old = lst[i]
                best_new = old
                best_s = s
                # Try random candidates
                for _ in range(12):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    d = delta_score(lst, m, old, new)
                    new_s = s + d
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                # Also try structured candidates: values that create new APs
                # Check values near existing elements
                for j in range(N):
                    for offset in [-2, -1, 1, 2]:
                        new = lst[j] + offset
                        if 1 <= new <= MAX_VAL and new != old and new not in s_set:
                            d = delta_score(lst, m, old, new)
                            new_s = s + d
                            if new_s > best_s:
                                best_s = new_s
                                best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << old) ^ (1 << best_new)
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    # Initialize
    pop_size = 35
    population = gen_initial_population(pop_size)
    population.sort(key=lambda x: x[2], reverse=True)
    best_individual = population[0]
    best_score = best_individual[2]
    best_list = best_individual[0][:]
    best_mask = best_individual[1]

    print(f"Initial best: {best_score}", file=sys.stderr)

    # GA loop
    generation = 0
    no_improve = 0
    elite_count = 4

    while time.time() < deadline - 20:
        generation += 1
        if generation % 8 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)

        new_population = []

        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])

        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 15:
                break
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            r = random.random()
            if r < 0.25:
                child = crossover(p1, p2)
            elif r < 0.6:
                child = mutate(p1, rate=0.12)
            else:
                # Simulated annealing-style: mutate with higher rate
                child = mutate(p1, rate=0.25)

            # Local search
            child = local_search(child, max_iter=60)
            new_population.append(child)

        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)

        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1

        # Restart if stuck
        if no_improve > 25:
            # Inject fresh diverse individuals
            for _ in range(6):
                # Multi-scale random
                base = random.randint(1, 30)
                step = random.randint(3, 15)
                lst = []
                for i in range(N):
                    v = base + i * step + random.randint(-2, 2)
                    if 1 <= v <= MAX_VAL:
                        lst.append(v)
                # Deduplicate and fill
                lst2 = list(dict.fromkeys(lst))
                sset = set(lst2)
                while len(lst2) < N:
                    v = random.randint(1, MAX_VAL)
                    if v not in sset:
                        sset.add(v)
                        lst2.append(v)
                m = list_to_mask(lst2[:N])
                s = score_mask(m)
                population.append((lst2[:N], m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0

    # Final exhaustive local search on best
    if time.time() < deadline - 3:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                old = lst[i]
                best_new = old
                best_s = s
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    d = delta_score(lst, m, old, new)
                    new_s = s + d
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << old) ^ (1 << best_new)
                    s = best_s
                    improved = True
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list

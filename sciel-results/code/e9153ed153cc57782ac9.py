"""Combined program: bitmask delta scoring + simulated annealing + diverse seeds."""
import time
import random

# EVOLVE-BLOCK-START

N = 32
MAX_VAL = 512
MAX_D = 255  # max d such that a + 2*d <= 512 for some a >= 1


def _list_to_mask(lst):
    m = 0
    for x in lst:
        m |= (1 << (x - 1))
    return m


def _mask_to_list(mask):
    return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]


def _score_mask(mask):
    count = 0
    for d in range(1, MAX_D + 1):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count


def _delta_score(mask, old_bit, new_bit):
    """Compute change in score when removing old_bit and adding new_bit."""
    new_mask = mask ^ old_bit ^ new_bit
    delta = 0
    for d in range(1, MAX_D + 1):
        old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
        new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
        if old_has != new_has:
            delta += 1 if new_has else -1
    return delta


def _gen_structured_seeds():
    """Generate diverse structured initial solutions."""
    solutions = []

    # 1. Pure arithmetic progressions
    for d in range(2, 17):
        if 1 + (N - 1) * d <= MAX_VAL:
            lst = [1 + i * d for i in range(N)]
            m = _list_to_mask(lst)
            s = _score_mask(m)
            solutions.append((lst, m, s))

    # 2. Sumset A + B constructions
    random.seed(42)
    for mult in [2, 3, 4, 5, 6, 7, 8]:
        for trial in range(40):
            sizeA = random.randint(2, 12)
            sizeB = random.randint(2, 12)
            A = set(random.sample(range(1, 100), min(sizeA, 99)))
            maxB = (MAX_VAL - 1) // mult
            B = set(random.sample(range(1, maxB + 1), min(sizeB, maxB)))
            S = set()
            for a in A:
                for b in B:
                    v = a + mult * b
                    if 1 <= v <= MAX_VAL:
                        S.add(v)
            if len(S) >= N:
                lst = random.sample(sorted(S), N)
                m = _list_to_mask(lst)
                s = _score_mask(m)
                solutions.append((lst, m, s))

    # 3. Residue classes mod m
    for mod in range(3, 15):
        for start in range(1, mod + 1):
            lst = []
            for v in range(start, MAX_VAL + 1, mod):
                lst.append(v)
                if len(lst) == N:
                    break
            if len(lst) == N:
                m = _list_to_mask(lst)
                s = _score_mask(m)
                solutions.append((lst, m, s))

    # 4. Random sets
    for _ in range(40):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        m = _list_to_mask(lst)
        s = _score_mask(m)
        solutions.append((lst, m, s))

    # 5. Greedy construction: build set element by element
    for _ in range(25):
        m = 0
        s = 0
        used = set()
        order = list(range(1, MAX_VAL + 1))
        random.shuffle(order)
        for v in order:
            if len(used) == N:
                break
            new_m = m | (1 << (v - 1))
            if new_m.bit_count() != m.bit_count() + 1:
                continue
            new_s = _score_mask(new_m)
            if new_s >= s:
                m = new_m
                s = new_s
                used.add(v)
        if len(used) == N:
            lst = list(used)
            m = _list_to_mask(lst)
            s = _score_mask(m)
            solutions.append((lst, m, s))

    # 6. Two-AP unions with varied parameters
    for d1 in range(2, 12):
        for d2 in range(2, 12):
            if d1 == d2:
                continue
            for n1 in range(4, 29):
                n2 = N - n1
                a1 = random.randint(1, 20)
                S1 = [a1 + i * d1 for i in range(n1)]
                if max(S1) > MAX_VAL:
                    continue
                a2 = random.randint(1, 20)
                S2 = [a2 + i * d2 for i in range(n2)]
                if max(S2) > MAX_VAL:
                    continue
                S = S1 + S2
                if len(set(S)) != N:
                    continue
                m = _list_to_mask(S)
                s = _score_mask(m)
                solutions.append((S, m, s))

    return solutions


def _simulated_annealing(individual, time_budget=15.0):
    """SA with delta scoring."""
    lst = individual[0][:]
    m = individual[1]
    s = individual[2]
    s_set = set(lst)
    best_lst = lst[:]
    best_m = m
    best_s = s

    start_sa = time.time()
    T = 10.0
    cooling = 0.995
    min_T = 0.1

    while time.time() - start_sa < time_budget:
        # Pick a random element to swap
        i = random.randint(0, N - 1)
        old = lst[i]
        old_bit = 1 << (old - 1)

        # Pick a random new value
        new = random.randint(1, MAX_VAL)
        if new == old or new in s_set:
            continue
        new_bit = 1 << (new - 1)

        delta = _delta_score(m, old_bit, new_bit)

        # Accept if improvement, or with probability exp(delta/T) if worse
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            # Accept move
            s_set.discard(old)
            s_set.add(new)
            lst[i] = new
            m ^= old_bit ^ new_bit
            s += delta

            if s > best_s:
                best_s = s
                best_lst = lst[:]
                best_m = m

        T *= cooling
        if T < min_T:
            T = min_T

    return (best_lst, best_m, best_s)


def _local_search(individual, time_budget=5.0):
    """Exhaustive single-swap local search using delta scoring."""
    lst = individual[0][:]
    m = individual[1]
    s = individual[2]
    s_set = set(lst)
    start_ls = time.time()

    improved = True
    while improved and time.time() - start_ls < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_ls >= time_budget:
                break
            old = lst[i]
            old_bit = 1 << (old - 1)
            best_new = old
            best_delta = 0
            for new in range(1, MAX_VAL + 1):
                if new == old or new in s_set:
                    continue
                new_bit = 1 << (new - 1)
                delta = _delta_score(m, old_bit, new_bit)
                if delta > best_delta:
                    best_delta = delta
                    best_new = new
            if best_new != old:
                s_set.discard(old)
                s_set.add(best_new)
                lst[i] = best_new
                m ^= old_bit ^ (1 << (best_new - 1))
                s += best_delta
                improved = True

    return (lst, m, s)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    def time_left():
        return deadline - time.time()

    # Generate diverse seeds
    seeds = _gen_structured_seeds()

    # Sort seeds by score, take top candidates
    seeds.sort(key=lambda x: x[2], reverse=True)
    top_seeds = seeds[:20]

    best_S = top_seeds[0][0]
    best_score = top_seeds[0][2]

    # Phase 1: Simulated annealing on top seeds
    for seed in top_seeds[:10]:
        if time_left() < 15:
            break
        sa_budget = min(8.0, time_left() - 10)
        if sa_budget < 2:
            break
        result = _simulated_annealing(seed, time_budget=sa_budget)
        if result[2] > best_score:
            best_score = result[2]
            best_S = result[0]

    # Phase 2: Local search polish on the best
    if time_left() > 5:
        best_ind = (best_S, _list_to_mask(best_S), best_score)
        ls_budget = min(10.0, time_left() - 3)
        if ls_budget > 1:
            result = _local_search(best_ind, time_budget=ls_budget)
            if result[2] > best_score:
                best_score = result[2]
                best_S = result[0]

    # Phase 3: Another round of SA from the polished result if time remains
    if time_left() > 8:
        best_ind = (best_S, _list_to_mask(best_S), best_score)
        sa_budget = min(10.0, time_left() - 3)
        if sa_budget > 2:
            result = _simulated_annealing(best_ind, time_budget=sa_budget)
            if result[2] > best_score:
                best_score = result[2]
                best_S = result[0]

    # Final local search
    if time_left() > 3:
        best_ind = (best_S, _list_to_mask(best_S), best_score)
        ls_budget = max(1.0, time_left() - 2)
        result = _local_search(best_ind, time_budget=ls_budget)
        if result[2] > best_score:
            best_score = result[2]
            best_S = result[0]

    return list(best_S)

# EVOLVE-BLOCK-END

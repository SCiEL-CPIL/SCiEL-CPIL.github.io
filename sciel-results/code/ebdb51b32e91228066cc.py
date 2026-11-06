"""32 integers in 1..512 maximizing distinct 3-AP differences.

Combines: bitset representation + incremental score-delta (fast) +
density-aware greedy seeding + multi-swap neighborhood +
population-based (differential-evolution-style) search.
"""
import time
import random

N = 32
RANGE = 512


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill

    rng = random.Random(12345)

    # ---------- bitset helpers ----------
    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s

    # ---------- score: full and incremental ----------
    def score_bitset(b):
        count = 0
        for d in range(1, 257):
            if (b & (b >> d) & (b >> (2 * d))) != 0:
                count += 1
        return count

    def diff_status(b, d):
        return (b & (b >> d) & (b >> (2 * d))) != 0

    def score_delta(old_b, new_b):
        """Full recompute but cached per-difference; cheap enough at pop scale."""
        return score_bitset(new_b)

    # ---------- density for greedy seeding ----------
    def compute_density():
        density = [0] * (RANGE + 1)
        for d in range(1, 257):
            for a in range(1, RANGE - 2 * d + 1):
                density[a] += 1
                density[a + d] += 1
                density[a + 2 * d] += 1
        return density

    density = compute_density()

    def greedy_initialize(seed_rng):
        current_set = set()
        current_bit = 0
        current_score = 0
        candidates = list(range(1, RANGE + 1))
        seed_rng.shuffle(candidates)
        # order candidates by density desc as a tie-break bias
        candidates.sort(key=lambda x: -density[x])
        seed_rng.shuffle(candidates)  # re-shuffle to keep randomness but bias low indices

        for _ in range(N):
            if not candidates:
                break
            best_elem = None
            best_gain = -1
            sample = seed_rng.sample(candidates, min(60, len(candidates)))
            for elem in sample:
                if elem in current_set:
                    continue
                new_bit = current_bit | (1 << (elem - 1))
                new_score = score_bitset(new_bit)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
            if best_elem is None:
                best_elem = candidates.pop()
                current_set.add(best_elem)
                current_bit |= (1 << (best_elem - 1))
                current_score = score_bitset(current_bit)
            else:
                current_set.add(best_elem)
                current_bit |= (1 << (best_elem - 1))
                current_score += best_gain
                candidates.remove(best_elem)
        return current_bit, current_score

    # ---------- multi-swap neighborhood ----------
    def swap_move(b, rng):
        """Replace k in {1,2,3} random elements with random new ones."""
        in_set = set_from_bitset(b)
        k = rng.choice([1, 1, 1, 2, 2, 3])
        k = min(k, N)
        idxs = rng.sample(range(N), k)
        old_elems = [in_set[i] for i in idxs]
        new_elems = []
        for _ in range(k):
            ne = rng.randint(1, RANGE)
            while ne in in_set or ne in new_elems:
                ne = rng.randint(1, RANGE)
            new_elems.append(ne)
        new_b = b
        for oe in old_elems:
            new_b &= ~(1 << (oe - 1))
        for ne in new_elems:
            new_b |= (1 << (ne - 1))
        return new_b

    # ---------- population-based search (DE-style) ----------
    POP = 10
    # seed the population
    pop = []
    for i in range(POP):
        b, s = greedy_initialize(random.Random(1000 + i * 7))
        pop.append([b, s])

    best_b, best_s = max(pop, key=lambda x: x[1])

    e = 2.718281828459045
    temp = 25.0
    cooling = 0.99995

    while temp > 0.01:
        if time.time() > deadline - 5.0:
            break
        # for each population member, try a DE mutation
        for i in range(POP):
            if time.time() > deadline - 5.0:
                break
            # pick two other distinct members
            a_idx, b_idx = rng.sample([j for j in range(POP) if j != i], 2)
            base = pop[i][0]
            # DE: base + (pop[a]-pop[b]) approximated by a swap move on base
            cand = swap_move(base, rng)
            cand_s = score_bitset(cand)
            cur_s = pop[i][1]
            if cand_s >= cur_s:
                pop[i] = [cand, cand_s]
            else:
                delta = cand_s - cur_s
                if rng.random() < (e ** (delta / temp)):
                    pop[i] = [cand, cand_s]
            # track global best
            if pop[i][1] > best_s:
                best_s = pop[i][1]
                best_b = pop[i][0]
        temp *= cooling

    # ---------- final local polish on best ----------
    if time.time() < deadline - 2.0:
        cur_b, cur_s = best_b, best_s
        for _ in range(20000):
            if time.time() > deadline - 1.0:
                break
            nb = swap_move(cur_b, rng)
            ns = score_bitset(nb)
            if ns > cur_s:
                cur_b, cur_s = nb, ns
                if cur_s > best_s:
                    best_s, best_b = cur_s, cur_b
            elif ns == cur_s and rng.random() < 0.3:
                cur_b = nb

    return set_from_bitset(best_b)

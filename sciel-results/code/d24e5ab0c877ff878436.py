import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    def compute_score(u):
        """Compute score using NumPy for speed."""
        u_arr = np.array(sorted(set(u)), dtype=np.int64)
        n = len(u_arr)
        if n < 2:
            return 1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def is_golomb(u):
        u = sorted(u)
        n = len(u)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    # ---- Bitset-based marginal scoring ----
    def popcount(x):
        return bin(x).count('1')

    def build_state(u):
        """Build bitset state for a set u."""
        u = sorted(set(u))
        n = len(u)
        if n == 0:
            return None
        max_u = max(u)
        sum_bits = 0
        diff_bits = 0
        for a in u:
            for b in u:
                sum_bits |= (1 << (a + b))
                diff_bits |= (1 << (a - b + max_u))
        count_sum = popcount(sum_bits)
        count_diff = popcount(diff_bits)
        return (u, sum_bits, diff_bits, count_sum, count_diff, max_u)

    def score_from_state(state):
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if count_sum == 0 or count_diff == 0:
            return 1.0
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        r = count_diff / count_sum
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def marginal_add(state, new_val):
        """Compute the new state after adding new_val, and the new score."""
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if new_val in u:
            return None
        new_max = max(max_u, new_val)
        new_sum_bits = 0
        for a in u:
            new_sum_bits |= (1 << (new_val + a))
        new_diff_bits = 0
        for a in u:
            new_diff_bits |= (1 << (new_val - a + new_max))
            new_diff_bits |= (1 << (a - new_val + new_max))

        if new_max != max_u:
            diff_bits = diff_bits << (new_max - max_u)

        combined_sum = sum_bits | new_sum_bits
        combined_diff = diff_bits | new_diff_bits
        new_count_sum = popcount(combined_sum)
        new_count_diff = popcount(combined_diff)

        new_u = sorted(u + [new_val])
        new_state = (new_u, combined_sum, combined_diff, new_count_sum, new_count_diff, new_max)
        new_score = score_from_state(new_state)
        return new_state, new_score

    def marginal_remove(state, val):
        """Compute the new state after removing val."""
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if val not in u or val == 0:
            return None
        new_u = [x for x in u if x != val]
        if not new_u:
            return None
        new_max = max(new_u)
        ns = build_state(new_u)
        if ns is None:
            return None
        new_score = score_from_state(ns)
        return ns, new_score

    # ---- Local search using bitsets ----
    def bitset_local_search(initial_u, time_budget=15.0):
        """Hill-climbing local search using bitset marginal evaluation."""
        start_ls = time.time()
        state = build_state(list(initial_u))
        if state is None:
            return list(initial_u)
        best_score = score_from_state(state)
        best_u = list(state[0])
        no_improve_count = 0

        while (time.time() - start_ls) < time_budget and remaining() > 5:
            improved = False
            u = state[0]
            max_u = state[5]

            # Try insertions: candidates from 1 to max_u + some margin
            candidates = []
            for v in range(1, max_u + len(u) * 3 + 10):
                if v not in u:
                    candidates.append(v)
            # Also try some random larger values
            for _ in range(20):
                v = random.randint(1, max_u + 50)
                if v not in u:
                    candidates.append(v)

            best_candidate = None
            best_candidate_score = best_score

            for v in candidates:
                if (time.time() - start_ls) > time_budget or remaining() < 5:
                    break
                result = marginal_add(state, v)
                if result is None:
                    continue
                new_state, new_score = result
                if new_score > best_candidate_score + 1e-12:
                    best_candidate_score = new_score
                    best_candidate = new_state

            # Try deletions
            for v in u:
                if v == 0:
                    continue
                if (time.time() - start_ls) > time_budget or remaining() < 5:
                    break
                result = marginal_remove(state, v)
                if result is None:
                    continue
                new_state, new_score = result
                if new_score > best_candidate_score + 1e-12:
                    best_candidate_score = new_score
                    best_candidate = new_state

            if best_candidate is not None:
                state = best_candidate
                best_score = best_candidate_score
                best_u = list(state[0])
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Try a random perturbation to escape local optimum
                if no_improve_count >= 3:
                    u = list(state[0])
                    if len(u) > 2:
                        # Randomly shift one element
                        idx = random.randint(1, len(u) - 1)
                        old_val = u[idx]
                        new_val = random.randint(1, max(u) + 30)
                        if new_val not in u:
                            u[idx] = new_val
                            u.sort()
                            new_state = build_state(u)
                            if new_state is not None:
                                new_score = score_from_state(new_state)
                                if new_score >= best_score - 0.02:
                                    state = new_state
                                    best_u = list(state[0])
                                    no_improve_count = 0
                    else:
                        no_improve_count = 0
                        v = random.randint(1, max(u) + 10)
                        if v not in u:
                            u.append(v)
                            u.sort()
                            new_state = build_state(u)
                            if new_state is not None:
                                state = new_state
                                best_u = list(state[0])
                                no_improve_count = 0

        return best_u

    # ---- Prime-field-inspired construction ----
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def construct_from_prime_field():
        """Use prime field multiplicative subgroup as seed."""
        seeds = []
        for p in [17, 31, 41, 61, 71, 101, 131, 151, 191, 211]:
            if not is_prime(p):
                continue
            for d in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                if (p - 1) % d != 0:
                    continue
                # Find a primitive root of Z_p*
                g = 2
                while True:
                    x = 1
                    order = 0
                    for _ in range(p - 1):
                        x = (x * g) % p
                        order += 1
                    if order == p - 1:
                        break
                    g += 1
                step = (p - 1) // d
                subgroup = [(g ** (step * k)) % p for k in range(d)]
                u = sorted(set([0] + subgroup))
                seeds.append(u)
        return seeds

    # ---- Known Golomb rulers as seeds ----
    known_rulers = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }

    valid_rulers = []
    for n, ruler in known_rulers.items():
        if is_golomb(ruler):
            valid_rulers.append(ruler)

    # ---- Generate candidates from multiple sources ----
    candidates = list(valid_rulers)
    pf_seeds = construct_from_prime_field()
    candidates.extend(pf_seeds)

    # Also try a few random initial sets
    for _ in range(3):
        if remaining() < 10:
            break
        n = random.randint(5, 12)
        u = [0] + sorted(random.sample(range(1, 150), n - 1))
        candidates.append(u)

    # ---- Run bitset local search on each candidate ----
    scored_candidates = []
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        scored_candidates.append((s, u))
    scored_candidates.sort(key=lambda x: -x[0])

    for s, u in scored_candidates:
        if remaining() < 8:
            break
        budget = min(10.0, remaining() * 0.12)
        if budget < 2.0:
            break
        improved_u = bitset_local_search(u, time_budget=budget)
        if improved_u and improved_u != u:
            candidates.append(improved_u)

    # ---- Find the best candidate ----
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u

    # ---- Final refinement on the best set ----
    if remaining() > 5 and len(best_set) >= 4:
        refined = bitset_local_search(best_set, time_budget=min(15.0, remaining() * 0.6))
        if refined:
            s = compute_score(refined)
            if s > best_score:
                best_set = refined
                best_score = s

    return best_set

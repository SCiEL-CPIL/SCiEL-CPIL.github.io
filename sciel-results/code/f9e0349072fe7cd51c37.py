import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    total = q ** n

    # Precompute neighbors (radius-1 ball)
    neigh, _ = _precompute_neighbors(q, n, radius)
    neigh_set = [set(x) for x in neigh]

    # Precompute pair distances for quick checks
    # dist[a][b] = Hamming distance
    # We'll compute on the fly but cache frequently used ones
    # Actually for 729 words, let's precompute all pairs
    dist = [[0] * total for _ in range(total)]
    for i in range(total):
        for j in range(i + 1, total):
            d = 0
            a, b = i, j
            for k in range(n):
                if a % q != b % q:
                    d += 1
                a //= q
                b //= q
            dist[i][j] = d
            dist[j][i] = d

    # Greedy initialization: pick word covering most uncovered, tie-break by distance to existing
    def greedy_init(rng):
        counts = [0] * total
        code = []
        in_code = set()
        rem_count = total

        def add(c):
            nonlocal rem_count
            code.append(c)
            in_code.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    rem_count -= 1

        while rem_count > 0:
            if time.monotonic() > deadline - 5.0:
                break
            best_c = -1
            best_gain = -1
            best_min_dist = -1

            # Sample a subset if too slow, but 729 is small
            for c in range(total):
                if c in in_code:
                    continue
                gain = 0
                for u in neigh[c]:
                    if counts[u] == 0:
                        gain += 1
                if gain <= best_gain:
                    continue
                # Compute min distance to existing code
                if not code:
                    md = n + 1
                else:
                    md = n + 1
                    for oc in code:
                        d = dist[c][oc]
                        if d < md:
                            md = d
                            if md <= 1:
                                break
                if gain > best_gain or (gain == best_gain and md > best_min_dist):
                    best_gain = gain
                    best_c = c
                    best_min_dist = md

            if best_c == -1:
                # Cover remaining arbitrarily
                for u in range(total):
                    if counts[u] == 0:
                        add(u)
                        break
                break
            add(best_c)

        return code, counts

    # Local search with simulated annealing
    def local_search(code, counts, rng, time_limit):
        temp = 1.0
        cool_rate = 0.9998
        min_temp = 0.005
        best_size = len(code)
        best_code = code[:]
        in_code = set(code)
        start_time = time.monotonic()

        def is_removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True

        def is_valid_swap(old_c, new_c):
            for u in neigh[old_c]:
                if counts[u] == 1 and u not in neigh_set[new_c]:
                    return False
            return True

        while time.monotonic() < start_time + time_limit and time.monotonic() < deadline - 2.0:
            if temp < min_temp:
                temp = min_temp

            r = rng.random()

            if r < 0.4:
                # Remove
                if code:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        in_code.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]

            elif r < 0.85:
                # Swap
                if code:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    if new_c == old_c or new_c in in_code:
                        continue
                    if is_valid_swap(old_c, new_c):
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        in_code.discard(old_c)
                        in_code.add(new_c)
                        code[idx] = new_c
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
            else:
                # Add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in in_code:
                    if rng.random() < math.exp(-1.5 / temp):
                        code.append(new_c)
                        in_code.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1

            temp *= cool_rate

        return best_size, best_code

    # Multiple restarts
    overall_best_size = total
    overall_best_code = None

    num_restarts = 6
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10.0:
            break

        rng = random.Random(1234 + restart * 7919)

        # Greedy init
        code, counts = greedy_init(rng)

        if overall_best_code is None or len(code) < overall_best_size:
            overall_best_size = len(code)
            overall_best_code = code[:]

        # Local search with remaining time
        remaining = deadline - time.monotonic() - 3.0
        ls_time = max(1.0, remaining / (num_restarts - restart))
        best_size, best_code = local_search(code, counts, rng, ls_time)

        if best_size < overall_best_size:
            overall_best_size = best_size
            overall_best_code = best_code

    # Final cleanup: remove any redundant words
    if overall_best_code is not None and time.monotonic() < deadline - 2.0:
        counts_final = [0] * total
        for c in overall_best_code:
            for u in neigh[c]:
                counts_final[u] += 1
        for i in range(len(overall_best_code) - 1, -1, -1):
            c = overall_best_code[i]
            removable = True
            for u in neigh[c]:
                if counts_final[u] == 1:
                    removable = False
                    break
            if removable:
                overall_best_code.pop(i)
                for u in neigh[c]:
                    counts_final[u] -= 1

    # Convert indices to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        t = idx
        for i in range(n):
            word[i] = t % q
            t //= q
        result.append(word)

    return result

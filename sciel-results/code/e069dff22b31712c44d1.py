import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin before 60s limit

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict graph
    # a*b % (a+b) == 0 iff: let g = gcd(a,b), a' = a/g, b' = b/g
    # then a+b = g*(a'+b'), a*b = g^2 * a'*b'
    # a*b / (a+b) = g * a' * b' / (a'+b')
    # So we need (a'+b') | (g * a' * b')
    # Since gcd(a', b') = 1, we need (a'+b') | g (because gcd(a'+b', a'*b') might not be 1)
    # Actually: gcd(a'+b', a') = gcd(b', a') = 1, and gcd(a'+b', b') = gcd(a', b') = 1
    # So gcd(a'+b', a'*b') = 1, hence we need (a'+b') | g
    conflict_mask = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            a, b = i, j
            # gcd
            aa, bb = a, b
            while bb:
                aa, bb = bb, aa % bb
            g = aa
            ap = a // g
            bp = b // g
            if g % (ap + bp) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def can_add(v, s_mask):
        return (conflict_mask[v] & s_mask) == 0

    def popcount(mask):
        return bin(mask).count('1')

    def mask_to_set(mask):
        return [v for v in vertices if (mask >> v) & 1]

    # Greedy: add vertices in given order if they don't conflict with current set
    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            if (result_mask >> v) & 1:
                continue
            if can_add(v, result_mask):
                result_mask |= (1 << v)
        return result_mask

    # Degree-based ordering
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(conflict_mask[i]).count('1')

    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]

    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
    ]

    # Random restarts
    for _ in range(30):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))

    best_mask = 0
    best_size = 0
    for c in candidates:
        sz = popcount(c)
        if sz > best_size:
            best_size = sz
            best_mask = c

    # Local search: add, 1-for-1 swap, 1-for-2 swap
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding each excluded vertex
            excluded = [v for v in vertices if not (s_mask >> v) & 1]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << v)
                    improved = True

            # Phase 2: 1-for-1 swaps
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    # Try to add a vertex that conflicts with u but not with rest
                    for v in vertices:
                        if v == u or (s_mask >> v) & 1:
                            continue
                        if can_add(v, temp_mask):
                            s_mask = temp_mask | (1 << v)
                            improved = True
                            break

            # Phase 3: 1-for-2 swaps (remove one, add two non-conflicting)
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    addable = []
                    for v in vertices:
                        if v != u and not (s_mask >> v) & 1 and can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 30)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if not ((conflict_mask[v1] >> v2) & 1):
                                    s_mask = temp_mask | (1 << v1) | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break

            # Phase 4: 2-for-3 swaps
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                limit_u = min(len(s_list), 20)
                for i in range(limit_u):
                    if time.time() >= time_limit:
                        break
                    for j in range(i + 1, limit_u):
                        if time.time() >= time_limit:
                            break
                        u1, u2 = s_list[i], s_list[j]
                        temp_mask = s_mask & ~(1 << u1) & ~(1 << u2)
                        addable = []
                        for v in vertices:
                            if v != u1 and v != u2 and not (s_mask >> v) & 1 and can_add(v, temp_mask):
                                addable.append(v)
                        if len(addable) >= 3:
                            random.shuffle(addable)
                            found = False
                            lim = min(len(addable), 15)
                            for a in range(lim):
                                if time.time() >= time_limit:
                                    break
                                for b in range(a + 1, lim):
                                    if time.time() >= time_limit:
                                        break
                                    for c in range(b + 1, lim):
                                        v1, v2, v3 = addable[a], addable[b], addable[c]
                                        if not ((conflict_mask[v1] >> v2) & 1) and \
                                           not ((conflict_mask[v1] >> v3) & 1) and \
                                           not ((conflict_mask[v2] >> v3) & 1):
                                            s_mask = temp_mask | (1 << v1) | (1 << v2) | (1 << v3)
                                            improved = True
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break

        return s_mask

    time_limit_ls1 = deadline - 30.0
    best_mask = local_search(best_mask, time_limit_ls1)
    best_size = popcount(best_mask)

    # Simulated annealing
    def simulated_annealing(s_mask, time_limit):
        current_mask = s_mask
        current_size = popcount(current_mask)
        best_mask_sa = current_mask
        best_size_sa = current_size

        t0 = 100.0
        t_end = 0.01
        cooling = 0.997
        temp = t0
        no_improve = 0

        while time.time() < time_limit and temp > t_end:
            move_type = random.random()

            if move_type < 0.30:
                # Try adding a random excluded vertex
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not excluded:
                    temp *= cooling
                    continue
                v = random.choice(excluded)
                if can_add(v, current_mask):
                    new_mask = current_mask | (1 << v)
                    new_size = current_size + 1
                    delta = 1
                else:
                    # Try 1-for-1: remove a conflict, add v
                    conflicts = [u for u in vertices if (current_mask >> u) & 1 and ((conflict_mask[v] >> u) & 1)]
                    if not conflicts:
                        temp *= cooling
                        continue
                    u = random.choice(conflicts)
                    temp_mask = current_mask & ~(1 << u)
                    if can_add(v, temp_mask):
                        new_mask = temp_mask | (1 << v)
                        new_size = current_size
                        delta = 0
                    else:
                        temp *= cooling
                        continue
            elif move_type < 0.50:
                # 1-for-2: remove one, add two
                included = [v for v in vertices if (current_mask >> v) & 1]
                if not included:
                    temp *= cooling
                    continue
                u = random.choice(included)
                temp_mask = current_mask & ~(1 << u)
                addable = [v for v in vertices if v != u and not (current_mask >> v) & 1 and can_add(v, temp_mask)]
                if len(addable) >= 2:
                    random.shuffle(addable)
                    lim = min(len(addable), 15)
                    found = False
                    for i in range(lim):
                        for j in range(i + 1, lim):
                            v1, v2 = addable[i], addable[j]
                            if not ((conflict_mask[v1] >> v2) & 1):
                                new_mask = temp_mask | (1 << v1) | (1 << v2)
                                new_size = current_size + 1
                                delta = 1
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        temp *= cooling
                        continue
                else:
                    temp *= cooling
                    continue
            elif move_type < 0.70:
                # 1-for-1 swap
                included = [v for v in vertices if (current_mask >> v) & 1]
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not included or not excluded:
                    temp *= cooling
                    continue
                u = random.choice(included)
                v = random.choice(excluded)
                temp_mask = current_mask & ~(1 << u)
                if can_add(v, temp_mask):
                    new_mask = temp_mask | (1 << v)
                    new_size = current_size
                    delta = 0
                else:
                    temp *= cooling
                    continue
            elif move_type < 0.85:
                # 2-for-3 swap
                included = [v for v in vertices if (current_mask >> v) & 1]
                if len(included) < 2:
                    temp *= cooling
                    continue
                random.shuffle(included)
                u1, u2 = included[0], included[1]
                temp_mask = current_mask & ~(1 << u1) & ~(1 << u2)
                addable = [v for v in vertices if v != u1 and v != u2 and not (current_mask >> v) & 1 and can_add(v, temp_mask)]
                if len(addable) >= 3:
                    random.shuffle(addable)
                    lim = min(len(addable), 12)
                    found = False
                    for a in range(lim):
                        for b in range(a + 1, lim):
                            for c in range(b + 1, lim):
                                v1, v2, v3 = addable[a], addable[b], addable[c]
                                if not ((conflict_mask[v1] >> v2) & 1) and \
                                   not ((conflict_mask[v1] >> v3) & 1) and \
                                   not ((conflict_mask[v2] >> v3) & 1):
                                    new_mask = temp_mask | (1 << v1) | (1 << v2) | (1 << v3)
                                    new_size = current_size + 1
                                    delta = 1
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        temp *= cooling
                        continue
                else:
                    temp *= cooling
                    continue
            else:
                # Escape: remove a random vertex
                included = [v for v in vertices if (current_mask >> v) & 1]
                if not included:
                    temp *= cooling
                    continue
                v = random.choice(included)
                new_mask = current_mask & ~(1 << v)
                new_size = current_size - 1
                delta = -1

            # Accept/reject
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size_sa:
                    best_mask_sa = current_mask
                    best_size_sa = current_size
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if random.random() < math.exp(delta / temp):
                    current_mask = new_mask
                    current_size = new_size
                no_improve += 1

            temp *= cooling

            if no_improve > 500:
                break

        return best_mask_sa

    sa_time_limit = deadline - 15.0
    sa_result = simulated_annealing(best_mask, sa_time_limit)
    if popcount(sa_result) > best_size:
        best_size = popcount(sa_result)
        best_mask = sa_result

    # Final local search polish
    if time.time() < deadline - 8.0:
        best_mask = local_search(best_mask, deadline - 5.0)
        best_size = popcount(best_mask)

    # Exhaustive addability check
    if time.time() < deadline - 3.0:
        for v in vertices:
            if not (best_mask >> v) & 1:
                if can_add(v, best_mask):
                    best_mask |= (1 << v)

    # Final validation
    result_list = sorted(mask_to_set(best_mask))
    if not result_list:
        return list(range(1, 257, 2))
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))

    return result_list

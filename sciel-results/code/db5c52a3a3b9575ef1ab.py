import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192

    def time_left():
        return deadline - (time.time() - start_time)

    def is_valid_set(S):
        """Check if S is valid: no element divides sum of 3 distinct others."""
        S = list(S)
        for idx, a in enumerate(S):
            others = [v for i, v in enumerate(S) if i != idx]
            L_len = len(others)
            if L_len < 3:
                continue

            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1

            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True

    def check_add(x, S):
        """Check if adding x to S keeps validity."""
        if x in S:
            return False
        S_list = list(S) + [x]
        return is_valid_set(S_list)

    def compute_conflict_count(x, S):
        """Count how many constraints would be violated if x were added to S."""
        conflicts = 0
        S_list = list(S)
        L = len(S_list)

        # Check: does x divide sum of any 3 distinct elements in S?
        if L >= 3:
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % x] += 1

            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % x
                    cnt = rem_count.get(target_rem, 0)
                    if u % x == target_rem:
                        cnt -= 1
                    if v % x == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        conflicts += cnt

        # Check: for each b in S, does b divide x + u + v for some distinct u,v in S\{b}?
        for b in S_list:
            if b == x:
                continue
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue

            target_rem = (-x) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1

            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    conflicts += cnt

        return conflicts

    # Precompute static "badness" for each number: how many triples from [1,n] sum to 0 mod d
    static_degree = [0] * (n + 1)
    for d in range(1, n + 1):
        cnt_res = [0] * d
        for v in range(1, n + 1):
            if v != d:
                cnt_res[v % d] += 1
        total = 0
        for r1 in range(d):
            for r2 in range(d):
                r3 = (-r1 - r2) % d
                total += cnt_res[r1] * cnt_res[r2] * cnt_res[r3]
        static_degree[d] = total

    # Strategy: Use a simulated annealing / random restart approach
    # with a strong initial solution from greedy, then try to improve.

    def greedy_construct(order):
        """Greedy: add elements in given order if valid."""
        S = []
        S_set = set()
        for x in order:
            if len(S) >= max_elements:
                break
            if x in S_set:
                continue
            if check_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, max_iters=500):
        """Try to improve S by swapping elements."""
        S = list(S)
        S_set = set(S)
        best = len(S)
        best_S = S[:]

        for it in range(max_iters):
            if time_left() < 5:
                break
            improved = False

            # Try to add one element
            if len(S) < max_elements:
                for x in range(1, n + 1):
                    if x in S_set:
                        continue
                    if check_add(x, S_set):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        break

            if not improved and len(S) > 1:
                # Try swap: remove one, add one
                for i in range(len(S)):
                    if time_left() < 5:
                        break
                    removed = S[i]
                    S_temp = [v for j, v in enumerate(S) if j != i]
                    S_set_temp = S_set - {removed}

                    for x in range(1, n + 1):
                        if x in S_set_temp:
                            continue
                        if check_add(x, S_set_temp):
                            # This swap maintains size but changes composition
                            # Only accept if it enables further additions
                            S_new = S_temp + [x]
                            S_set_new = set(S_new)
                            # Try to add more
                            can_add_more = False
                            for y in range(1, n + 1):
                                if y in S_set_new:
                                    continue
                                if check_add(y, S_set_new):
                                    can_add_more = True
                                    break
                            if can_add_more:
                                S = S_new
                                S_set = S_set_new
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                # Try 2-for-1 swap: remove 2, add 3
                if len(S) >= 2 and len(S) + 1 < max_elements:
                    for i in range(len(S)):
                        if time_left() < 5:
                            break
                        for j in range(i + 1, len(S)):
                            if time_left() < 5:
                                break
                            S_temp = [v for k, v in enumerate(S) if k != i and k != j]
                            S_set_temp = set(S_temp)

                            # Try to add 3 elements
                            added = []
                            for x in range(1, n + 1):
                                if len(added) >= 3:
                                    break
                                if x in S_set_temp:
                                    continue
                                if check_add(x, S_set_temp | set(added)):
                                    added.append(x)
                                    S_set_temp.add(x)

                            if len(added) >= 3:
                                S = S_temp + added
                                S_set = set(S)
                                improved = True
                                break
                        if improved:
                            break

            if not improved:
                break

            if len(S) > best:
                best = len(S)
                best_S = S[:]

        return best_S

    def simulated_annealing(S, max_iters=2000):
        """Simulated annealing to improve the set."""
        S = list(S)
        S_set = set(S)
        best = len(S)
        best_S = S[:]
        T = 100.0
        cooling = 0.9995

        for it in range(max_iters):
            if time_left() < 5:
                break

            # Generate a move
            move_type = random.random()
            if len(S) < max_elements and move_type < 0.3:
                # Add an element
                x = random.randint(1, n)
                if x in S_set:
                    continue
                if check_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best:
                        best = len(S)
                        best_S = S[:]
                else:
                    # Accept with probability exp(-delta/T) where delta is conflict count
                    conf = compute_conflict_count(x, S)
                    if conf > 0:
                        prob = min(1.0, 1.0 / (1.0 + conf))
                        if random.random() < prob:
                            S.append(x)
                            S_set.add(x)
            elif len(S) > 1 and move_type < 0.6:
                # Remove an element
                i = random.randint(0, len(S) - 1)
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
            elif len(S) > 0 and move_type < 0.8:
                # Swap
                i = random.randint(0, len(S) - 1)
                x = random.randint(1, n)
                if x in S_set:
                    continue
                removed = S[i]
                S_temp = [v for j, v in enumerate(S) if j != i]
                S_set_temp = S_set - {removed}
                if check_add(x, S_set_temp):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best:
                        best = len(S)
                        best_S = S[:]
            else:
                # 2-for-1 or 1-for-2
                if len(S) >= 2:
                    i = random.randint(0, len(S) - 1)
                    j = random.randint(0, len(S) - 1)
                    if i == j:
                        continue
                    S_temp = [v for k, v in enumerate(S) if k != i and k != j]
                    S_set_temp = set(S_temp)
                    # Try to add 2-3 random elements
                    added = []
                    for _ in range(random.randint(2, 3)):
                        x = random.randint(1, n)
                        if x in S_set_temp or x in added:
                            continue
                        if check_add(x, S_set_temp | set(added)):
                            added.append(x)
                            S_set_temp.add(x)
                    if len(added) >= 2:
                        S = S_temp + added
                        S_set = set(S)
                        if len(S) > best:
                            best = len(S)
                            best_S = S[:]

            T *= cooling

        return best_S

    # Main algorithm:
    # 1. Generate multiple initial solutions using different greedy orders
    # 2. Improve each with local search and simulated annealing
    # 3. Return the best found

    best_S = []
    best_size = 0

    # Generate diverse initial orders
    orders = []

    # Order 1: Descending (large numbers first - less likely to divide sums)
    orders.append(list(range(n, 0, -1)))

    # Order 2: Ascending
    orders.append(list(range(1, n + 1)))

    # Order 3: By static degree (low to high)
    orders.append(sorted(range(1, n + 1), key=lambda x: static_degree[x]))

    # Order 4: By static degree (high to low)
    orders.append(sorted(range(1, n + 1), key=lambda x: -static_degree[x]))

    # Order 5: Random
    for _ in range(5):
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    # Order 6: Prime-like (numbers with few divisors)
    def count_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt

    orders.append(sorted(range(1, n + 1), key=lambda x: count_divisors(x)))
    orders.append(sorted(range(1, n + 1), key=lambda x: -count_divisors(x)))

    # Order 7: Mix of large and small
    large = list(range(n, n // 2, -1))
    small = list(range(1, n // 4 + 1))
    mixed = large + small
    random.shuffle(mixed)
    orders.append(mixed)

    # Run greedy for each order
    for order in orders:
        if time_left() < 10:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]

    # Improve best with local search
    if time_left() > 15:
        improved = local_search(best_S, max_iters=300)
        if len(improved) > best_size:
            best_size = len(improved)
            best_S = improved[:]

    # Try simulated annealing on best
    if time_left() > 15:
        sa_result = simulated_annealing(best_S, max_iters=1000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_S = sa_result[:]

    # Try more random restarts if time permits
    restart_count = 0
    while time_left() > 20 and restart_count < 10:
        restart_count += 1
        # Random initial solution
        random_set = set()
        for _ in range(random.randint(20, 40)):
            x = random.randint(1, n)
            if x not in random_set and check_add(x, random_set):
                random_set.add(x)

        if len(random_set) > best_size:
            best_size = len(random_set)
            best_S = list(random_set)

        # Improve with local search
        if time_left() > 10:
            improved = local_search(list(random_set), max_iters=200)
            if len(improved) > best_size:
                best_size = len(improved)
                best_S = improved[:]

    # Final local search on best
    if time_left() > 5:
        final = local_search(best_S, max_iters=200)
        if len(final) > best_size:
            best_S = final

    return best_S

import time
import random

def construct():
    deadline = time.time() + 85
    N = 512
    target_size = 48

    def is_valid_full(S):
        m = len(S)
        if m < 3:
            return True
        for a in S:
            others = [x for x in S if x != a]
            max_k = len(others)
            dp = [0] * (max_k + 1)
            dp[0] = 1
            count = 0
            for v in others:
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            for k in range(2, max_k + 1):
                target = a * k
                if target > 512 * 48:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def fast_pair_check(S, x):
        S_set = set(S)
        for a in S:
            b = 2 * x - a
            if b in S_set and b != a:
                return False
        return True

    def greedy_build(order):
        current = []
        current_set = set()
        for c in order:
            if len(current) >= target_size:
                break
            if c in current_set:
                continue
            if not fast_pair_check(current, c):
                continue
            test = current + [c]
            if is_valid_full(test):
                current.append(c)
                current_set.add(c)
        return current

    rng = random.Random(42)

    best_set = []
    best_len = 0

    # Deterministic greedy orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
    ]
    for order in orders:
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]

    # Randomized greedy
    for i in range(300):
        if time.time() > deadline - 15:
            break
        p = list(range(1, N + 1))
        rng.shuffle(p)
        s = greedy_build(p)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len >= target_size:
                return best_set[:target_size]

    # Local search with add and swap
    if best_len < target_size:
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            cur = best_set[:]
            cur_set = set(cur)
            candidates = [x for x in range(1, N + 1) if x not in cur_set]
            rng.shuffle(candidates)

            added = False
            for x in candidates:
                if len(cur) >= target_size:
                    break
                if not fast_pair_check(cur, x):
                    continue
                test = cur + [x]
                if is_valid_full(test):
                    cur.append(x)
                    cur_set.add(x)
                    added = True
                    if len(cur) > best_len:
                        best_len = len(cur)
                        best_set = cur[:]
                        if best_len >= target_size:
                            return best_set[:target_size]

            if not added and cur and candidates:
                # Swap
                a = rng.choice(cur)
                new_set = [v for v in cur if v != a]
                new_set_set = set(new_set)
                found = False
                for x in candidates:
                    if x in new_set_set:
                        continue
                    if not fast_pair_check(new_set, x):
                        continue
                    test = new_set + [x]
                    if is_valid_full(test):
                        cur = test
                        cur_set = set(cur)
                        found = True
                        # Fill up
                        remaining = [y for y in range(1, N + 1) if y not in cur_set]
                        rng.shuffle(remaining)
                        for y in remaining:
                            if len(cur) >= target_size:
                                break
                            if not fast_pair_check(cur, y):
                                continue
                            test2 = cur + [y]
                            if is_valid_full(test2):
                                cur.append(y)
                                cur_set.add(y)
                        if len(cur) > best_len:
                            best_len = len(cur)
                            best_set = cur[:]
                            if best_len >= target_size:
                                return best_set[:target_size]
                        break

    return best_set

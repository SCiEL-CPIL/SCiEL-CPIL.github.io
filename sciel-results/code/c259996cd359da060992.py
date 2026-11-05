import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def compute_sums(seq):
        L = len(seq)
        if L == 0:
            return set()
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums

    def is_valid(seq):
        L = len(seq)
        if L == 0:
            return True
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def incremental_check(seq, x, existing_sums):
        k = bisect.bisect_left(seq, x)
        L = len(seq)
        new_sums = set()
        if x in existing_sums:
            return False, None
        new_sums.add(x)
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            s = left_sum + x
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        right_sum = 0
        for j in range(k, L):
            right_sum += seq[j]
            s = x + right_sum
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            right_sum = 0
            for j in range(k, L):
                right_sum += seq[j]
                s = left_sum + x + right_sum
                if s in existing_sums or s in new_sums:
                    return False, None
                new_sums.add(s)
        return True, new_sums

    def greedy_build(strategy, existing_sums=None):
        seq = []
        sums = set() if existing_sums is None else existing_sums.copy()
        if strategy == 'increasing':
            candidates = list(range(1, 257))
        elif strategy == 'decreasing':
            candidates = list(range(256, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            candidates = strategy
        for x in candidates:
            if x in seq:
                continue
            valid, new_sums = incremental_check(seq, x, sums)
            if valid:
                k = bisect.bisect_left(seq, x)
                seq = seq[:k] + [x] + seq[k:]
                sums.update(new_sums)
        return seq, sums

    best_seq = []
    best_sums = set()

    strategies = ['increasing', 'decreasing']
    for _ in range(3):
        rand_strat = list(range(1, 257))
        random.shuffle(rand_strat)
        strategies.append(rand_strat)

    for strat in strategies:
        seq, sums = greedy_build(strat)
        if len(seq) > len(best_seq):
            best_seq = seq
            best_sums = sums

    # Position-aware insertion: try inserting x at every valid position
    def try_insert_all_positions(seq, x, existing_sums):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]

        for pos in range(L + 1):
            if pos > 0 and seq[pos - 1] >= x:
                continue
            if pos < L and x >= seq[pos]:
                continue
            new_sums = set()
            base_left = P[pos]
            conflict = False
            for l in range(0, pos + 1):
                left_part = base_left - P[pos - l]
                for r in range(0, L - pos + 1):
                    if l == 0 and r == 0:
                        continue
                    right_part = P[pos + r] - P[pos]
                    s = left_part + x + right_part
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.add(s)
                if conflict:
                    break
            if conflict:
                continue
            new_seq = seq[:pos] + [x] + seq[pos:]
            return new_seq, existing_sums | new_sums
        return None

    current_seq = list(best_seq)
    current_sums = set(best_sums)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)

    all_values = set(range(1, 257))
    current_set = set(current_seq)
    remaining = all_values - current_set

    # Backtracking DFS
    def backtracking_dfs(seq, sums, remaining, depth, max_depth):
        if depth >= max_depth or time.time() > deadline - 1:
            return list(seq), set(sums)
        best_seq = list(seq)
        best_sums = set(sums)
        for x in sorted(remaining, key=lambda v: abs(v - (seq[-1] if seq else 128))):
            valid, new_sums = incremental_check(seq, x, sums)
            if valid:
                k = bisect.bisect_left(seq, x)
                new_seq = seq[:k] + [x] + seq[k:]
                new_sums_set = sums | new_sums
                remaining.remove(x)
                deeper_seq, deeper_sums = backtracking_dfs(new_seq, new_sums_set, remaining, depth + 1, max_depth)
                remaining.add(x)
                if len(deeper_seq) > len(best_seq):
                    best_seq = deeper_seq
                    best_sums = deeper_sums
        return best_seq, best_sums

    if time.time() < deadline - 5 and len(remaining) > 0:
        dfs_seq, dfs_sums = backtracking_dfs(current_seq, current_sums, set(remaining), 0, 2)
        if len(dfs_seq) > best_len:
            best_len = len(dfs_seq)
            best_seq_final = list(dfs_seq)
            current_seq = dfs_seq
            current_sums = dfs_sums
            current_set = set(current_seq)
            remaining = all_values - current_set

    # Local search with position-aware insertion
    stuck_count = 0
    max_stuck = 50

    while time.time() < deadline - 1:
        r = random.random()

        if r < 0.55:
            # Add with position-aware insertion
            if remaining:
                available = list(remaining)
                # Heuristic: try a few candidates, prefer ones that are spaced
                tried = 0
                max_tries = min(5, len(available))
                while tried < max_tries:
                    x = random.choice(available)
                    available.remove(x)
                    tried += 1
                    result = try_insert_all_positions(current_seq, x, current_sums)
                    if result is not None:
                        new_seq, new_sums = result
                        current_seq = new_seq
                        current_sums = new_sums
                        remaining.remove(x)
                        stuck_count = 0
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq_final = list(current_seq)
                        break
                else:
                    stuck_count += 1

        elif r < 0.75:
            # Remove a random element
            if current_seq and len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                remaining.add(x_rem)
                current_sums = compute_sums(current_seq)
                stuck_count = 0

        elif r < 0.9:
            # Swap: remove one, try to add another at any position
            if current_seq and remaining:
                x_rem = random.choice(current_seq)
                x_add = random.choice(list(remaining))
                temp_seq = [x for x in current_seq if x != x_rem]
                result = try_insert_all_positions(temp_seq, x_add, compute_sums(temp_seq))
                if result is not None:
                    new_seq, new_sums = result
                    current_seq = new_seq
                    current_sums = new_sums
                    remaining.remove(x_add)
                    stuck_count = 0
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq_final = list(current_seq)
                else:
                    stuck_count += 1

        else:
            # Fresh greedy restart
            rand_strat = list(range(1, 257))
            random.shuffle(rand_strat)
            seq, sums = greedy_build(rand_strat)
            if len(seq) > len(current_seq):
                current_seq = seq
                current_sums = sums
                current_set = set(current_seq)
                remaining = all_values - current_set
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_final = list(current_seq)
            stuck_count = 0

        if stuck_count > max_stuck:
            rand_strat = list(range(1, 257))
            random.shuffle(rand_strat)
            seq, sums = greedy_build(rand_strat)
            if len(seq) > len(current_seq):
                current_seq = seq
                current_sums = sums
                current_set = set(current_seq)
                remaining = all_values - current_set
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_final = list(current_seq)
            stuck_count = 0

    return sorted(best_seq_final)

import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # safety margin before 60s kill
    MAX_VAL = 256

    def get_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_block_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def can_extend(seq, new_val, existing_sums):
        n = len(seq)
        new_sums = []
        s = new_val
        new_sums.append(s)
        for i in range(n - 1, -1, -1):
            s += seq[i]
            new_sums.append(s)
        for ns in new_sums:
            if ns in existing_sums:
                return False
        if len(new_sums) != len(set(new_sums)):
            return False
        return True

    def greedy_asc(start=1, step=1):
        vals = []
        sums = set()
        for x in range(start, MAX_VAL + 1, step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        return vals

    def greedy_desc(start=MAX_VAL, step=1):
        vals = []
        sums = set()
        for x in range(start, 0, -step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        vals.sort()
        return vals

    def try_extend(seq, time_limit):
        current = list(seq)
        used = set(current)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for pos in range(len(current) + 1):
                if pos == 0:
                    lo, hi = 1, (current[0] - 1 if current else MAX_VAL)
                elif pos == len(current):
                    lo, hi = current[-1] + 1, MAX_VAL
                else:
                    lo, hi = current[pos - 1] + 1, current[pos] - 1
                if lo > hi:
                    continue
                for x in range(hi, lo - 1, -1):
                    if x in used:
                        continue
                    new_seq = current[:pos] + [x] + current[pos:]
                    new_sums = get_block_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return current

    # ---- SIMULATED ANNEALING on the set of selected indices ----
    # State: a strictly increasing sequence (sorted list of chosen values).
    # Moves: (a) insert a random value into a random gap, (b) remove a random element,
    # (c) swap a random element with a random other value.
    # Objective: maximize length; penalty for invalidity.
    def simulated_annealing(init_seq, time_budget, rng):
        def energy(seq):
            # Lower is better. We want max length, and validity.
            if not is_valid(seq):
                # Penalize invalid: large penalty + negative length
                return 10000 - len(seq)
            return -len(seq)

        current = list(init_seq)
        current_e = energy(current)
        best = list(current)
        best_e = current_e

        T_start = 50.0
        T_end = 0.01
        # Number of steps we can afford
        steps = 0
        deadline = time.time() + time_budget

        while time.time() < deadline:
            steps += 1
            # Annealing schedule
            progress = min(1.0, steps / max(1, int((time_budget) * 200)))
            T = T_start * (T_end / T_start) ** progress

            n = len(current)
            move_type = rng.random()
            
            new_seq = list(current)

            if move_type < 0.4:
                # Insert a random value
                if n == 0:
                    new_seq = [rng.randint(1, MAX_VAL)]
                else:
                    pos = rng.randint(0, n)
                    if pos == 0:
                        lo, hi = 1, (current[0] - 1 if current else MAX_VAL)
                    elif pos == n:
                        lo, hi = current[-1] + 1, MAX_VAL
                    else:
                        lo, hi = current[pos - 1] + 1, current[pos] - 1
                    if lo <= hi:
                        x = rng.randint(lo, hi)
                        new_seq = current[:pos] + [x] + current[pos:]
                    else:
                        continue
            elif move_type < 0.7:
                # Remove a random element
                if n > 1:
                    idx = rng.randint(0, n - 1)
                    new_seq = current[:idx] + current[idx+1:]
                else:
                    continue
            else:
                # Swap: remove one, insert a different value elsewhere
                if n > 0:
                    idx = rng.randint(0, n - 1)
                    old_val = current[idx]
                    # Choose a new value
                    new_val = rng.randint(1, MAX_VAL)
                    # Build candidate
                    temp = current[:idx] + current[idx+1:]
                    # Insert new_val into a valid position
                    placed = False
                    for pos in range(len(temp) + 1):
                        if pos == 0:
                            lo, hi = 1, (temp[0] - 1 if temp else MAX_VAL)
                        elif pos == len(temp):
                            lo, hi = temp[-1] + 1, MAX_VAL
                        else:
                            lo, hi = temp[pos - 1] + 1, temp[pos] - 1
                        if lo <= new_val <= hi:
                            new_seq = temp[:pos] + [new_val] + temp[pos:]
                            placed = True
                            break
                    if not placed:
                        continue

            new_e = energy(new_seq)
            delta = new_e - current_e

            if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-9)):
                current = new_seq
                current_e = new_e
                if current_e < best_e:
                    best = list(current)
                    best_e = current_e

        return best

    # ---- Build diverse candidate pool ----
    candidates = []
    candidates.append(greedy_asc())
    candidates.append(greedy_desc())
    for step in (2, 3, 4, 5):
        candidates.append(greedy_asc(step=step))
        candidates.append(greedy_desc(step=step))

    # Add some random starts
    rng = random.Random(42)
    for _ in range(10):
        start = rng.randint(1, 50)
        candidates.append(greedy_asc(start=start))
    for _ in range(10):
        start = rng.randint(200, MAX_VAL)
        candidates.append(greedy_desc(start=start))

    best_seq = max(candidates, key=len)
    best_len = len(best_seq)

    # Refinement pass on the best candidates
    for seq in sorted(candidates, key=len, reverse=True):
        if time.time() > TIME_LIMIT - 3.0:
            break
        if len(seq) < best_len - 3:
            continue
        ext = try_extend(list(seq), TIME_LIMIT - 3.0)
        if len(ext) > best_len:
            best_seq, best_len = ext, len(ext)

    # Simulated annealing on the best and a few near-best candidates
    for seq in sorted(candidates, key=len, reverse=True)[:5]:
        if time.time() > TIME_LIMIT - 2.0:
            break
        sa_result = simulated_annealing(list(seq), TIME_LIMIT - 2.0, rng)
        if len(sa_result) > best_len:
            best_seq, best_len = sa_result, len(sa_result)

    # Final extension of the global best
    ext = try_extend(list(best_seq), TIME_LIMIT - 1.0)
    if len(ext) > best_len:
        best_seq = ext

    # Final verification
    if is_valid(best_seq):
        return best_seq
    else:
        # Fallback: return a known valid sequence
        return [1, 2, 4, 8, 16, 32, 64, 128]

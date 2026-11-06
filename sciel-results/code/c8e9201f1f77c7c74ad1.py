import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0

    def compute_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def all_sums_distinct(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Greedy init
    best_seq = []
    used_sums = set()
    for x in range(1, 257):
        new_sums = set()
        s = x
        new_sums.add(s)
        for a in reversed(best_seq):
            s += a
            new_sums.add(s)
        if not new_sums & used_sums:
            best_seq.append(x)
            used_sums |= new_sums

    best_len = len(best_seq)
    if best_len == 0:
        return [1]

    # Simulated annealing
    cur = list(best_seq)
    rng = random.Random(42)
    T = 1.0
    T_min = 0.05
    cool = 0.9995
    best_sa_len = best_len
    best_sa_seq = list(best_seq)

    while time.time() - start_time < TIME_LIMIT:
        T = max(T * cool, T_min)
        r = rng.random()

        if r < 0.3:
            # Add a random element
            avail = [i for i in range(1, 257) if i not in cur]
            if not avail:
                continue
            x = rng.choice(avail)
            pos = 0
            for i in range(len(cur)):
                if cur[i] < x:
                    pos = i + 1
                else:
                    break
            new = cur[:pos] + [x] + cur[pos:]
            if all_sums_distinct(new):
                nl = len(new)
                d = nl - best_sa_len
                if d >= 0 or rng.random() < 2.718281828 ** (d / T):
                    cur = new
                    if nl > best_sa_len:
                        best_sa_len = nl
                        best_sa_seq = list(cur)

        elif r < 0.6:
            # Remove a random element
            if not cur:
                continue
            idx = rng.randrange(len(cur))
            new = cur[:idx] + cur[idx + 1:]
            if all_sums_distinct(new):
                nl = len(new)
                d = nl - best_sa_len
                if d >= 0 or rng.random() < 2.718281828 ** (d / T):
                    cur = new
                    if nl > best_sa_len:
                        best_sa_len = nl
                        best_sa_seq = list(cur)

        else:
            # Swap: remove one, add another
            if not cur:
                continue
            idx = rng.randrange(len(cur))
            rm = cur[idx]
            avail = [i for i in range(1, 257) if i not in cur]
            if not avail:
                continue
            add = rng.choice(avail)
            tmp = cur[:idx] + cur[idx + 1:]
            pos = 0
            for i in range(len(tmp)):
                if tmp[i] < add:
                    pos = i + 1
                else:
                    break
            new = tmp[:pos] + [add] + tmp[pos:]
            if all_sums_distinct(new):
                nl = len(new)
                d = nl - best_sa_len
                if d >= 0 or rng.random() < 2.718281828 ** (d / T):
                    cur = new
                    if nl > best_sa_len:
                        best_sa_len = nl
                        best_sa_seq = list(cur)

    if best_sa_len > best_len:
        return best_sa_seq
    return best_seq

import random
import time

N = 16
DEADLINE = 110.0


def construct():
    T0 = time.time()

    def left():
        return DEADLINE - (time.time() - T0)

    rng = random.Random(98765)

    all_triples = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                all_triples.append((a, b, c))
    all_set = set(all_triples)

    # Forbidden 8-sets
    forbidden = []
    seen = set()
    for x in range(N):
        for y in range(x + 1, N):
            for z in range(y + 1, N):
                for w in range(z + 1, N):
                    for u in range(w + 1, N):
                        for v in range(u + 1, N):
                            pts = (x, y, z, w, u, v)
                            if pts in seen:
                                continue
                            seen.add(pts)
                            f8 = set()
                            for i in range(2):
                                for j in range(2):
                                    for k in range(2):
                                        f8.add((pts[i * 2], pts[j * 2], pts[k * 2]))
                            forbidden.append(f8)

    def can_add(s, t):
        for f in forbidden:
            if f <= (s | {t}):
                return False
        return True

    def can_remove(s, t):
        for f in forbidden:
            if f <= s and t in f:
                return False
        return True

    # Greedy
    def greedy():
        s = set()
        order = all_triples[:]
        rng.shuffle(order)
        for t in order:
            if t not in s and can_add(s, t):
                s.add(t)
        return s

    best = 105
    best_triples = set((0, a, b) for a in range(1, N) for b in range(a + 1, N))

    def local_search(s):
        s = set(s)
        cur = len(s)
        improved = True
        stall = 0
        while improved and left() > 5:
            improved = False
            stall += 1
            if stall > 60:
                break
            missing = [t for t in all_triples if t not in s]
            cur_list = list(s)
            rng.shuffle(cur_list)
            rng.shuffle(missing)
            for rem in cur_list:
                if left() < 5:
                    break
                s.discard(rem)
                for add in missing:
                    if add == rem or add in s:
                        continue
                    if can_add(s, add):
                        s.add(add)
                        if len(s) > cur:
                            cur = len(s)
                            improved = True
                            stall = 0
                        break
                if improved:
                    break
        return s

    for restart in range(8):
        if left() < 12:
            break
        s = greedy()
        s = local_search(s)
        if len(s) > best:
            best = len(s)
            best_triples = s

    return [list(t) for t in best_triples]

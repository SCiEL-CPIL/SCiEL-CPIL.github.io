import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    def elapsed():
        return time.time() - t0

    n = 256
    weight = [0.0] * (n + 1)
    for x in range(1, n + 1):
        weight[x] = 1.0 / x

    def blocked_of(T):
        """Numbers x whose addition to T would complete a 3-AP with two elements of T."""
        B = set()
        lst = list(T)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                c1 = 2 * a - b
                if c1 != a and c1 != b:
                    B.add(c1)
                c2 = 2 * b - a
                if c2 != a and c2 != b:
                    B.add(c2)
                s = a + b
                if (s & 1) == 0:
                    mid = s >> 1
                    if mid != a and mid != b:
                        B.add(mid)
        return B

    def best_addable(T, B):
        """Smallest x (largest weight) addable to T given blocked set B."""
        for x in range(1, n + 1):
            if x not in T and x not in B:
                return x
        return None

    def complete_maximal(S):
        """Greedily add best available until no valid addition remains."""
        S = set(S)
        while True:
            if elapsed() > DEADLINE:
                break
            B = blocked_of(S)
            x = best_addable(S, B)
            if x is None:
                break
            S.add(x)
        return S

    def value(S):
        return sum(weight[x] for x in S)

    # ---------- seeds ----------
    def seed_cantor():
        S = set()
        def gen(p, pw):
            if p > n:
                return
            if pw == 0:
                if p >= 1:
                    S.add(p)
                return
            gen(p, pw - 1)
            gen(p + pw, pw - 1)
        gen(0, 9)
        return S

    def seed_greedy(asc=True):
        S = set()
        order = range(1, n + 1) if asc else range(n, 0, -1)
        for x in order:
            if x not in S and best_addable(S, blocked_of(S)) is not None:
                # quick check: is x addable?
                B = blocked_of(S)
                if x not in B:
                    S.add(x)
        return S

    def seed_random(rng):
        S = set()
        lst = list(range(1, n + 1))
        rng.shuffle(lst)
        for x in lst:
            B = blocked_of(S)
            if x not in B:
                S.add(x)
        return S

    # ---------- local search ----------
    def improve(S, K=40):
        S = complete_maximal(S)
        v = value(S)
        while elapsed() < DEADLINE:
            B = blocked_of(S)
            add_x = best_addable(S, B)
            best_dv = 1e-12
            best_move = None
            # plain add
            if add_x is not None:
                best_dv = weight[add_x]
                best_move = (add_x, None)
            # remove a large blocker, then add best newly available
            if S:
                big = sorted(S, key=lambda z: -z)[:K]  # largest elements first
                for a in big:
                    S2 = S - {a}
                    B2 = blocked_of(S2)
                    b = best_addable(S2, B2)
                    if b is None:
                        continue
                    dv = weight[b] - weight[a]
                    if dv > best_dv:
                        best_dv = dv
                        best_move = (b, a)
            if best_move is None or best_dv <= 1e-12:
                break
            b, a = best_move
            if a is not None:
                S.discard(a)
            S.add(b)
            v += weight[b] - (weight[a] if a is not None else 0.0)
        return S, v

    best_S = None
    best_val = -1.0

    rng = random.Random(987654321)
    seeds = [seed_cantor(), seed_greedy(True), seed_greedy(False)]
    for _ in range(10):
        seeds.append(seed_random(rng))

    for S in seeds:
        if elapsed() > DEADLINE:
            break
        S2, v = improve(S)
        if v > best_val:
            best_val = v
            best_S = set(S2)

    if not best_S:
        best_S = {1}
    return list(best_S)

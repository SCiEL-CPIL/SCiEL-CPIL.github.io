import time
import random
from itertools import combinations
from collections import Counter

N = 192
MAXK = 48
ALL = list(range(1, N + 1))


class FeasibleSet:
    """Maintains a set S and fast structures to test add(x) in O(|S|^2).

    conflict: element a is bad iff a | (sum of 3 distinct others).
    We precompute:
      triple_sums[s] = # triples {b,c,d} in S with b+c+d = s
      pair_mod[a][r] = # pairs {b,c} in S\{a} with (b+c) % a == r
    Adding x:
      - x-bad: any triple (b,c,d) in S with b+c+d == k*x
      - a-bad (a in S): any pair (b,c) in S\{a} with (b+c) % a == (-x) % a
    """

    def __init__(self):
        self.S = []
        self.Sset = set()
        self.triple_sums = Counter()
        self.pair_mod = {}
        self._min3 = None
        self._max3 = None

    def __len__(self):
        return len(self.S)

    def rebuild(self, seq):
        S = list(seq)
        Sset = set(S)
        self.S = S
        self.Sset = Sset
        # triple sums
        ts = Counter()
        for b, c, d in combinations(Sset, 3):
            ts[b + c + d] += 1
        self.triple_sums = ts
        # pair mod per element
        pm = {}
        for a in Sset:
            cnt = Counter()
            for b, c in combinations(Sset - {a}, 2):
                cnt[(b + c) % a] += 1
            pm[a] = cnt
        self.pair_mod = pm
        self._update_extremes()

    def _update_extremes(self):
        if len(self.S) >= 3:
            ss = sorted(self.S)
            self._min3 = ss[0] + ss[1] + ss[2]
            self._max3 = ss[-1] + ss[-2] + ss[-3]
        else:
            self._min3 = None
            self._max3 = None

    def can_add(self, x):
        S = self.S
        Sset = self.Sset
        if x in Sset:
            return False
        # x-bad: triple sum that is a multiple of x within [min3, max3]
        if self._max3 is not None:
            ts = self.triple_sums
            min3 = self._min3
            max3 = self._max3
            k = (min3 + x - 1) // x
            while k * x <= max3:
                if ts[k * x] > 0:
                    return False
                k += 1
        # a-bad for a in S
        pm = self.pair_mod
        for a in S:
            if pm[a][(-x) % a] > 0:
                return False
        return True

    def add(self, x):
        Sset = self.Sset
        S = self.S
        ts = self.triple_sums
        pm = self.pair_mod
        # update triple sums: x + pairs in S
        for b, c in combinations(Sset, 2):
            ts[x + b + c] += 1
        # update pair_mod for existing a (a != x, x not in S so always true)
        for a in S:
            cnt = pm[a]
            for b in S:
                if b != a:
                    cnt[(x + b) % a] += 1
        # new entry for x
        cntx = Counter()
        for b, c in combinations(Sset, 2):
            cntx[(b + c) % x] += 1
        pm[x] = cntx
        S.append(x)
        Sset.add(x)
        self._update_extremes()

    def remove(self, x):
        Sset = self.Sset
        S = self.S
        ts = self.triple_sums
        pm = self.pair_mod
        for b, c in combinations(Sset - {x}, 2):
            ts[x + b + c] -= 1
        for a in S:
            if a == x:
                continue
            cnt = pm[a]
            for b in S:
                if b != a and b != x:
                    cnt[(x + b) % a] -= 1
        del pm[x]
        S.remove(x)
        Sset.discard(x)
        self._update_extremes()

    def snapshot(self):
        return (list(self.S), dict(self.triple_sums),
                {a: dict(c) for a, c in self.pair_mod.items()},
                self._min3, self._max3)

    def restore(self, snap):
        S, ts, pm, min3, max3 = snap
        self.S = S
        self.Sset = set(S)
        self.triple_sums = Counter(ts)
        self.pair_mod = {a: Counter(c) for a, c in pm.items()}
        self._min3 = min3
        self._max3 = max3


def greedy_build(fs, order):
    for x in order:
        if len(fs) >= MAXK:
            break
        if fs.can_add(x):
            fs.add(x)
    return fs


def local_improve(fs, deadline, rng):
    """Try to grow via single adds; return improved set (mutates fs in place)."""
    improved = True
    while improved and len(fs) < MAXK and time.time() < deadline:
        improved = False
        cands = [x for x in ALL if x not in fs.Sset]
        rng.shuffle(cands)
        for x in cands:
            if len(fs) >= MAXK:
                break
            if fs.can_add(x):
                fs.add(x)
                improved = True
    return fs


def swap_improve(fs, deadline, rng, rounds=4):
    """1-for-1 swaps that keep size but allow later growth; accept if growth possible."""
    best_size = len(fs)
    for _ in range(rounds):
        if time.time() >= deadline or len(fs) >= MAXK:
            break
        progress = False
        order = list(ALL)
        rng.shuffle(order)
        for out in fs.S[:]:
            for inn in order:
                if len(fs) >= MAXK:
                    break
                if inn in fs.Sset:
                    continue
                # tentative remove then check add
                fs.remove(out)
                if fs.can_add(inn):
                    fs.add(inn)
                    # try to grow
                    local_improve(fs, deadline, rng)
                    if len(fs) > best_size:
                        best_size = len(fs)
                        progress = True
                        break
                    else:
                        # revert swap
                        fs.remove(inn)
                        fs.add(out)
                else:
                    fs.add(out)
        if not progress:
            break
    return fs


def construct():
    deadline = time.time() + 80  # safety margin under 90s
    rng = random.Random(20240517)

    # ---- initial greedy solutions with several orderings ----
    def order_desc():
        return list(range(N, 0, -1))

    def order_degree():
        # fewer divisors (in 1..N) = "harder to divide sums" -> prefer
        def dcount(x):
            c = 0
            for d in range(2, x + 1):
                if x % d == 0:
                    c += 1
            return c
        return sorted(ALL, key=lambda x: (dcount(x), -x))

    def order_rand(bias=0.0):
        return sorted(ALL, key=lambda x: rng.random() * 60 + x * bias)

    inits = [order_desc(), order_degree()]
    for _ in range(6):
        inits.append(order_rand(bias=rng.uniform(0.2, 1.0)))

    best_size = 0
    best_seq = []
    for ordi in inits:
        if time.time() >= deadline - 5:
            break
        fs = FeasibleSet()
        greedy_build(fs, ordi)
        local_improve(fs, deadline, rng)
        swap_improve(fs, deadline, rng)
        if len(fs) > best_size:
            best_size = len(fs)
            best_seq = list(fs.S)
        if best_size >= MAXK:
            break

    if not best_seq:
        # fallback: pure largest-first greedy without improvement (should not happen)
        fs = FeasibleSet()
        greedy_build(fs, order_desc())
        return list(fs.S)

    # ---- global improvement from the best set ----
    fs = FeasibleSet()
    fs.rebuild(best_seq)
    local_improve(fs, deadline, rng)
    swap_improve(fs, deadline, rng)
    if len(fs) > best_size:
        best_size = len(fs)
        best_seq = list(fs.S)

    # ---- simulated annealing over the fast feasibility oracle ----
    # objective: maximize size; secondary: prefer larger elements (more robust)
    def score(seq):
        return 100 * len(seq) + sum(seq)

    # work from a copy
    fs = FeasibleSet()
    fs.rebuild(best_seq)
    cur_score = score(fs.S)
    best_seq = list(fs.S)
    best_score = cur_score

    t0 = time.time()
    start = t0
    duration = max(1.0, deadline - t0 - 3)
    T0 = 1.0
    cool = 0.000004  # tuned small

    # precompute element "quality" for tie-breaking / diversity
    used_count = [0] * (N + 1)

    steps = 0
    while time.time() < deadline - 3:
        elapsed = time.time() - start
        T = max(1e-4, T0 - cool * elapsed)

        move = rng.random()
        fs.remove_last = None
        if len(fs) == 0:
            # must add
            x = rng.choice(ALL)
            if fs.can_add(x):
                fs.add(x)
                cur_score = score(fs.S)
                if cur_score > best_score:
                    best_score = cur_score
                    best_seq = list(fs.S)
        elif move < 0.45 and len(fs) < MAXK:
            # add
            cand = rng.choice(ALL)
            if cand in fs.Sset:
                continue
            if fs.can_add(cand):
                fs.add(cand)
                cur_score = score(fs.S)
                if cur_score > best_score:
                    best_score = cur_score
                    best_seq = list(fs.S)
        elif move < 0.80 and len(fs) > 1:
            # remove a random element
            x = fs.S[rng.randrange(len(fs.S))]
            fs.remove(x)
            cur_score = score(fs.S)
            if cur_score > best_score:
                best_score = cur_score
                best_seq = list(fs.S)
        else:
            # swap
            if len(fs) == 0:
                continue
            out = fs.S[rng.randrange(len(fs.S))]
            inn = rng.choice(ALL)
            if inn in fs.Sset:
                continue
            snap = fs.snapshot()
            fs.remove(out)
            if fs.can_add(inn):
                fs.add(inn)
                new_score = score(fs.S)
                delta = new_score - cur_score
                if delta >= 0 or (T > 1e-5 and rng.random() < min(1.0, delta / (T + 1e-9) * -1)):
                    # accept improving or (rarely) non-improving by random
                    if delta >= 0 or rng.random() < min(0.5, 1.0 / (1.0 + abs(delta))):
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_seq = list(fs.S)
                    else:
                        fs.restore(snap)
                else:
                    fs.restore(snap)
            else:
                fs.restore(snap)

        steps += 1
        if steps % 200 == 0:
            # periodic polish to lock in gains
            local_improve(fs, min(time.time() + 0.3, deadline - 3), rng)
            cur_score = score(fs.S)
            if cur_score > best_score:
                best_score = cur_score
                best_seq = list(fs.S)

    # ---- final polishing ----
    fs = FeasibleSet()
    fs.rebuild(best_seq)
    local_improve(fs, deadline, rng)
    if len(fs) > len(best_seq):
        best_seq = list(fs.S)

    return best_seq

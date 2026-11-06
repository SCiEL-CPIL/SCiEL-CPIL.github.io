import time
import random

def construct():
    start = time.time()
    deadline = start + 86  # hard wall-clock kill is 90s; keep a margin

    n = 192
    max_size = 48
    max_sum = 3 * n

    # --- Precomputation -------------------------------------------------------
    # divisors_of[s]: all d in [1,n] that divide s
    divisors_of = [[] for _ in range(max_sum + 1)]
    for s in range(1, max_sum + 1):
        for d in range(1, min(s, n) + 1):
            if s % d == 0:
                divisors_of[s].append(d)

    # bitset of present values: bit i set  <=>  i in S
    # (we keep this as an int; bit i = 1<<i)
    # precompute bit[i]
    bit = [1 << i for i in range(n + 1)]

    # For each candidate a, the set of forbidden-sum values: multiples of a in [3, max_sum]
    # Represented as a list of multiples.
    multiples_of = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        multiples_of[a] = list(range(a, max_sum + 1, a))

    # --- Sumset maintenance ---------------------------------------------------
    # sumset: bitset over [0..max_sum]; bit M set iff there exist 3 DISTINCT selected
    # elements summing to M.
    # We maintain it incrementally. When adding/removing an element x, we update
    # using the "pair-sum" bitset of the set WITHOUT x.
    # pairset: bitset over [0..2n]; bit t set iff there exist 2 DISTINCT selected
    # elements summing to t.

    class SetState:
        __slots__ = ('S', 'sumset', 'pairset', 'count')
        def __init__(self):
            self.S = set()
            self.sumset = 0
            self.pairset = 0
            self.count = 0

        def _pairset_without(self, x):
            # pairset of S \ {x}
            p = self.pairset
            # remove pairs that include x: for each y in S, y != x, pair (x+y)
            for y in self.S:
                if y != x:
                    p &= ~bit[x + y]
            return p

        def add(self, x):
            if x in self.S:
                return
            # new pairs formed by x with every existing y
            new_pairs = 0
            for y in self.S:
                new_pairs |= bit[x + y]
            # new triples formed by x + (pair in old set)
            old_pairset = self.pairset
            new_triples = 0
            # for each pair-sum t in old_pairset, t + x is a new triple sum
            # We iterate set bits of old_pairset
            pp = old_pairset
            while pp:
                low = pp & (-pp)
                t = low.bit_length() - 1
                m = t + x
                if m <= max_sum:
                    new_triples |= bit[m]
                pp ^= low
            self.pairset |= new_pairs
            self.sumset |= new_triples
            self.S.add(x)
            self.count += 1

        def remove(self, x):
            if x not in self.S:
                return
            # recompute pairset and sumset from scratch excluding x
            rest = self.S - {x}
            rp = 0
            rs = 0
            rl = list(rest)
            K = len(rl)
            # pairs
            for i in range(K):
                a = rl[i]
                for j in range(i + 1, K):
                    b = rl[j]
                    rp |= bit[a + b]
            # triples: for each pair (a,b), for each c > b, sum
            for i in range(K):
                a = rl[i]
                for j in range(i + 1, K):
                    b = rl[j]
                    ab = a + b
                    for k in range(j + 1, K):
                        c = rl[k]
                        m = ab + c
                        if m <= max_sum:
                            rs |= bit[m]
            self.S = rest
            self.pairset = rp
            self.sumset = rs
            self.count = K

    # --- Validity check for adding x to a state -------------------------------
    def can_add(x, st):
        # Condition A: x must not divide any sum of 3 distinct OTHER selected elements.
        # i.e., no multiple M of x (M >= 3) is in the sumset of S (which uses only S, not x).
        ss = st.sumset
        for M in multiples_of[x]:
            if M < 3:
                continue
            if (ss >> M) & 1:
                return False
        # Condition B: no b in S should divide (x + u + v) for distinct u,v in S,
        # with b != u, b != v (and b is in S, automatically b != x since x not in S).
        S = st.S
        # iterate pairs (u, v) in S
        Sl = list(S)
        K = len(Sl)
        for i in range(K):
            u = Sl[i]
            for j in range(i + 1, K):
                v = Sl[j]
                T = x + u + v
                if T > max_sum:
                    continue
                for b in divisors_of[T]:
                    if b == u or b == v:
                        continue
                    if b in S:
                        return False
        return True

    # --- Full validity of a set (for final verification / safety) -------------
    def is_valid(S):
        S = set(S)
        K = len(S)
        if K < 4:
            return True
        Sl = sorted(S)
        for idx, a in enumerate(Sl):
            others = [x for i, x in enumerate(Sl) if i != idx]
            # build sumset of others
            Lo = len(others)
            oset = set(others)
            for i in range(Lo):
                x = others[i]
                for j in range(i + 1, Lo):
                    y = others[j]
                    for k in range(j + 1, Lo):
                        z = others[k]
                        M = x + y + z
                        if M % a == 0:
                            return False
        return True

    # --- Greedy with a given order -------------------------------------------
    def greedy_build(order):
        st = SetState()
        for x in order:
            if st.count >= max_size:
                break
            if can_add(x, st):
                st.add(x)
        return st

    # --- Local search ---------------------------------------------------------
    def local_search(st, rng, budget_deadline):
        best = set(st.S)
        best_size = st.count
        cur = SetState()
        # copy
        for x in st.S:
            cur.add(x)
        all_numbers = set(range(1, n + 1))
        while time.time() < budget_deadline and cur.count < max_size:
            improved = False
            # 1) try to add a random element
            cand = list(all_numbers - cur.S)
            rng.shuffle(cand)
            for x in cand[:40]:
                if can_add(x, cur):
                    cur.add(x)
                    improved = True
                    if cur.count > best_size:
                        best_size = cur.count
                        best = set(cur.S)
                    break
            if improved:
                continue
            # 2) try 1-for-1 swap
            Sl = list(cur.S)
            rng.shuffle(Sl)
            swap = False
            for y in Sl[:25]:
                S2 = cur.S - {y}
                for x in cand[:40]:
                    if x in S2:
                        continue
                    # build a temp state = S2
                    tmp = SetState()
                    for z in S2:
                        tmp.add(z)
                    if can_add(x, tmp):
                        # commit: remove y from cur, add x
                        cur.remove(y)
                        cur.add(x)
                        swap = True
                        if cur.count > best_size:
                            best_size = cur.count
                            best = set(cur.S)
                        break
                if swap:
                    break
            if swap:
                continue
            # 3) try 1-for-2 (remove y, add two)
            swap2 = False
            for y in Sl[:15]:
                S2 = cur.S - {y}
                tmp = SetState()
                for z in S2:
                    tmp.add(z)
                c2 = list(all_numbers - S2)
                rng.shuffle(c2)
                for x1 in c2[:25]:
                    if can_add(x1, tmp):
                        tmp2 = SetState()
                        for z in S2:
                            tmp2.add(z)
                        tmp2.add(x1)
                        for x2 in c2[:25]:
                            if x2 == x1 or x2 in S2:
                                continue
                            if can_add(x2, tmp2):
                                cur.remove(y)
                                cur.add(x1)
                                cur.add(x2)
                                swap2 = True
                                if cur.count > best_size:
                                    best_size = cur.count
                                    best = set(cur.S)
                                break
                        if swap2:
                            break
                    if swap2:
                        break
                if swap2:
                    break
            if not swap2:
                break
        return best

    # --- Orders ---------------------------------------------------------------
    def get_orders(rng):
        orders = []
        orders.append(list(range(n, 0, -1)))
        orders.append(list(range(1, n + 1)))
        # primes first
        primes = [x for x in range(2, n + 1) if all(x % p != 0 for p in range(2, int(x ** 0.5) + 1))]
        non_primes = [x for x in range(1, n + 1) if x not in primes]
        orders.append(primes + non_primes)
        # large-first with a bit of randomization
        for _ in range(8):
            o = list(range(1, n + 1))
            rng.shuffle(o)
            o.sort(key=lambda x: -x + rng.random() * 20)
            orders.append(o)
        # random
        for _ in range(12):
            o = list(range(1, n + 1))
            rng.shuffle(o)
            orders.append(o)
        return orders

    # --- Main -----------------------------------------------------------------
    best_overall = set()
    best_size_overall = 0

    rng = random.Random(20240501)
    orders = get_orders(rng)

    # Pass 1: greedy initializations
    for o in orders:
        if time.time() > deadline - 18:
            break
        st = greedy_build(o)
        if st.count > best_size_overall:
            best_size_overall = st.count
            best_overall = set(st.S)

    # Pass 2: local search from the best greedy
    if time.time() < deadline - 6 and best_size_overall > 0:
        st0 = SetState()
        for x in best_overall:
            st0.add(x)
        improved = local_search(st0, rng, deadline - 2)
        if len(improved) > best_size_overall:
            best_size_overall = len(improved)
            best_overall = improved

    # Pass 3: local search from a few other greedy starts (only if close)
    for o in orders[:6]:
        if time.time() > deadline - 4:
            break
        st = greedy_build(o)
        if st.count >= best_size_overall - 2 and st.count > 0:
            improved = local_search(st, rng, deadline - 2)
            if len(improved) > best_size_overall:
                best_size_overall = len(improved)
                best_overall = improved

    # Safety: verify the final set is valid; if not, trim
    if best_overall and not is_valid(best_overall):
        # fallback: rebuild greedily descending
        st = greedy_build(list(range(n, 0, -1)))
        best_overall = st.S

    return list(best_overall)

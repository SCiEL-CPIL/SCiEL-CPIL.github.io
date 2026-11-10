"""Structural rewrite: model disjointness as per-prime residue-slot packing.

Two residue classes (r_i, m_i) and (r_j, m_j) are disjoint iff
gcd(m_i, m_j) does NOT divide (r_i - r_j).  Equivalently, for every prime
p dividing gcd(m_i, m_j) the residues must differ mod p.  Hence, for each
prime p, the selected moduli divisible by p must carry pairwise-distinct
residues mod p, i.e. at most p of them.  This is a packing problem: prime p
offers p "slots"; each modulus consumes one slot for every distinct prime
factor.  We maximize the number of selected moduli (distinct, in [2,128])
with a consistent residue assignment (always feasible via CRT since the
combined residue is < rad(m) <= m).
"""
import time
import math
import random

MAXM = 128


def _primes_upto(n):
    sieve = [True] * (n + 1)
    if n >= 0:
        sieve[0] = False
    if n >= 1:
        sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


_PRIMES = _primes_upto(MAXM)


def _distinct_primes(m):
    res = []
    for p in _PRIMES:
        if p * p > m:
            break
        if m % p == 0:
            res.append(p)
            while m % p == 0:
                m //= p
    if m > 1:
        res.append(m)
    return res


_FACTORS = {m: _distinct_primes(m) for m in range(2, MAXM + 1)}
_ALL_MODULI = list(range(2, MAXM + 1))


def _residue_for(m, chosen):
    """Given chosen[p] for each distinct prime p|m, return the unique r in
    [0, rad(m)) with r % p == chosen[p] (CRT).  Always < m."""
    rad = 1
    for p in _FACTORS[m]:
        rad *= p
    # small search; rad <= m <= 128 so this is tiny
    for r in range(rad):
        ok = True
        for p in _FACTORS[m]:
            if r % p != chosen[p]:
                ok = False
                break
        if ok:
            return r
    return 0  # unreachable


class _State:
    __slots__ = ("used", "free", "selected", "res_of")

    def __init__(self):
        self.used = {p: set() for p in _PRIMES}
        self.free = {p: p for p in _PRIMES}
        self.selected = []          # list of moduli
        self.res_of = {}            # modulus -> residue

    def can_add(self, m):
        for p in _FACTORS[m]:
            if self.free[p] <= 0:
                return False
        return True

    def add(self, m):
        chosen = {}
        for p in _FACTORS[m]:
            for v in range(p):
                if v not in self.used[p]:
                    chosen[p] = v
                    break
        r = _residue_for(m, chosen)
        for p in _FACTORS[m]:
            self.used[p].add(r % p)
            self.free[p] -= 1
        self.selected.append(m)
        self.res_of[m] = r
        return r

    def remove(self, m):
        r = self.res_of[m]
        for p in _FACTORS[m]:
            self.used[p].discard(r % p)
            self.free[p] += 1
        self.selected.remove(m)
        del self.res_of[m]

    def addable(self, exclude=None):
        out = []
        sel = set(self.selected)
        for m in _ALL_MODULI:
            if m in sel:
                continue
            if exclude is not None and m in exclude:
                continue
            if self.can_add(m):
                out.append(m)
        return out

    def result(self):
        return [[self.res_of[m], m] for m in self.selected]


def _greedy(order, deadline):
    st = _State()
    for m in order:
        if time.time() > deadline:
            break
        if st.can_add(m):
            st.add(m)
    return st


def _local_search(st, deadline):
    """1-for-2 improvement: drop one selected modulus, try to add two new
    ones.  Repeat until no gain or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(st.selected)
        for i in range(n):
            if time.time() > deadline:
                break
            a = st.selected[i]
            st.remove(a)
            addable = st.addable(exclude={a})
            if len(addable) >= 2:
                # prefer the two that consume the most "expensive" small primes
                addable.sort(key=lambda m: (min(_FACTORS[m]), m))
                b, c = addable[0], addable[1]
                st.add(b)
                st.add(c)
                improved = True
                # restart scan
                break
            else:
                st.add(a)  # restore
    return st


def _orders():
    orders = []
    # smallest prime factor descending (cheap large-prime moduli first)
    orders.append(sorted(_ALL_MODULI, key=lambda m: (-min(_FACTORS[m]), m)))
    # fewest distinct prime factors first
    orders.append(sorted(_ALL_MODULI, key=lambda m: (len(_FACTORS[m]), m)))
    # largest modulus first
    orders.append(sorted(_ALL_MODULI, key=lambda m: -m))
    # radical ascending
    def rad(m):
        r = 1
        for p in _FACTORS[m]:
            r *= p
        return r
    orders.append(sorted(_ALL_MODULI, key=lambda m: (rad(m), m)))
    # a few random orders
    rng = random.Random(12345)
    for _ in range(4):
        o = list(_ALL_MODULI)
        rng.shuffle(o)
        orders.append(o)
    return orders


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin vs 90s kill
    best = None
    best_len = -1

    for order in _orders():
        if time.time() > deadline:
            break
        st = _greedy(order, deadline)
        st = _local_search(st, deadline)
        if len(st.selected) > best_len:
            best_len = len(st.selected)
            best = st.result()
        # if we already exceed a strong bar, keep polishing the best only
        if best_len >= 24:
            break

    if best is None or best_len <= 0:
        # guaranteed nonempty fallback
        return [[1, 2]]
    return best


# EVOLVE-BLOCK-END

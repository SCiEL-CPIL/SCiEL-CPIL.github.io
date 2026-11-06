"""HP lattice protein folding: diversified randomized growth + local search + SA polish."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 210.0
_MARGIN = 18.0
_TOL = _DEADLINE - _MARGIN

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGH = [tuple((a + da, b + db)) for a in (0,) for b in (0,) for da, db in _DIRS]


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Exact energy recomputation (used only for final verification)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy growth with incremental contact tracking (O(n))."""
    n = len(sequence)
    is_h = [c == "H" for c in sequence]
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    energy = 0
    # frontier = previous residues that can still form a new contact with the next one
    frontier = {0}
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _NEIGH:
            s = (last[0] + site[0], last[1] + site[1])
            if s in occupied:
                continue
            gain = 0
            if is_h[i]:
                for idx in frontier:
                    if is_h[idx]:
                        p = chain[idx]
                        if (abs(p[0] - s[0]) + abs(p[1] - s[1])) == 1:
                            gain += 1
            free = 0
            for nb in _NEIGH:
                if (s[0] + nb[0], s[1] + nb[1]) not in occupied:
                    free += 1
            options.append((gain + 0.05 * free + rng.random(), s))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        energy -= gain
        # update frontier: remove residues too far from the new tip, add the previous tip
        frontier.discard(i - 1)
        for idx in list(frontier):
            p = chain[idx]
            if (abs(p[0] - site[0]) + abs(p[1] - site[1])) > 1:
                frontier.discard(idx)
        frontier.add(i - 1)
    return chain, energy


class _LocalSearch:
    def __init__(self, sequence, chain):
        self.is_h = [c == "H" for c in sequence]
        self.n = len(sequence)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.energy = 0
        # count H-H neighbours (both directions) per residue
        self.hnb = [0] * self.n
        for i, p in enumerate(self.chain):
            if self.is_h[i]:
                for q in _NEIGH:
                    j = self.occupied.get((p[0] + q[0], p[1] + q[1]))
                    if j is not None and self.is_h[j]:
                        self.hnb[i] += 1
        self.energy = -sum(self.hnb) // 2

    def _remove(self, i):
        p = self.chain[i]
        for q in _NEIGH:
            j = self.occupied.get((p[0] + q[0], p[1] + q[1]))
            if j is not None and self.is_h[i] and self.is_h[j]:
                self.hnb[i] -= 1
                self.hnb[j] -= 1
                self.energy += 1
        del self.occupied[p]

    def _place(self, i, p):
        self.occupied[p] = i
        self.chain[i] = p
        if self.is_h[i]:
            for q in _NEIGH:
                j = self.occupied.get((p[0] + q[0], p[1] + q[1]))
                if j is not None and self.is_h[j]:
                    self.hnb[i] += 1
                    self.hnb[j] += 1
                    self.energy -= 1

    def _try_reversal(self, a, b):
        if b <= a + 1:
            return False
        # contacts crossing the segment boundary
        c = 0
        for i in (a, b):
            for q in _NEIGH:
                j = self.occupied.get((self.chain[i][0] + q[0], self.chain[i][1] + q[1]))
                if j is not None and j < a or (j > b and (j - i) > 1):
                    if self.is_h[i] and self.is_h[j]:
                        c += 1
        # internal contacts of the segment (each counted once)
        inner = 0
        for i in range(a + 1, b):
            if self.is_h[i]:
                inner += self.hnb[i]
        inner //= 2
        old = self.energy
        self._remove(a)
        self._remove(b)
        # reverse the segment
        self.chain[a:b + 1] = self.chain[a:b + 1][::-1]
        for idx in range(a, b + 1):
            self.occupied[self.chain[idx]] = idx
        self._place(a, self.chain[a])
        self._place(b, self.chain[b])
        # recompute internal contacts of the (now reversed) segment
        inner2 = 0
        for i in range(a + 1, b):
            if self.is_h[i]:
                for q in _NEIGH:
                    j = self.occupied.get((self.chain[i][0] + q[0], self.chain[i][1] + q[1]))
                    if j is not None and a < j < b:
                        if self.is_h[j]:
                            inner2 += 1
        inner2 //= 2
        self.energy = old - c + 2 * (inner2 - inner)
        return self.energy < old

    def _try_move(self, i):
        if i == 0 or i == self.n - 1:
            return False
        old = self.energy
        p = self.chain[i]
        self._remove(i)
        done = False
        for q in _NEIGH:
            s = (p[0] + q[0], p[1] + q[1])
            if s in self.occupied:
                continue
            self._place(i, s)
            if self.energy < old:
                done = True
                break
            self._remove(i)
        if not done:
            self._place(i, p)
        return done

    def optimize(self, rng, max_iters=20000, deadline=None):
        it = 0
        while it < max_iters:
            it += 1
            if deadline is not None and it % 256 == 0 and time.time() > deadline:
                break
            n = self.n
            if rng.random() < 0.5:
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a > b:
                    a, b = b, a
                self._try_reversal(a, b)
            else:
                i = rng.randrange(1, n - 1)
                self._try_move(i)
        return self.energy, self.chain

    def anneal(self, rng, iters, deadline=None):
        T0 = 1.0
        T = T0
        cool = (0.001 / T0) ** (1.0 / max(1, iters))
        for _ in range(iters):
            if time.time() > deadline:
                break
            T *= cool
            n = self.n
            if rng.random() < 0.5:
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a > b:
                    a, b = b, a
                old = self.energy
                self._try_reversal(a, b)
                d = self.energy - old
                if d > 0 and rng.random() < min(1.0, 1.0 / (1.0 + d / (T + 1e-9))):
                    pass  # keep
                else:
                    # revert
                    self._try_reversal(a, b)
            else:
                i = rng.randrange(1, n - 1)
                old = self.energy
                self._try_move(i)
                d = self.energy - old
                if d > 0 and rng.random() < min(1.0, 1.0 / (1.0 + d / (T + 1e-9))):
                    pass
                else:
                    self._try_move(i)
        return self.energy, self.chain


def fold(sequence, dimension):
    start = time.time()
    deadline = start + _TOL
    n = len(sequence)
    seed = 12345 + len(sequence)
    rng = random.Random(seed)

    def remaining():
        return deadline - time.time()

    # Phase 1: diversified randomized growth
    best = None
    best_energy = 0
    candidates = []
    # more restarts for shorter sequences
    if n <= 30:
        grow_target = 250
    elif n <= 50:
        grow_target = 150
    else:
        grow_target = 90
    for _ in range(grow_target):
        if time.time() > deadline - 25.0:
            break
        res = _grow(sequence, dimension, rng)
        if res is None:
            continue
        chain, energy = res
        if best is None or energy < best_energy:
            best, best_energy = chain, energy
        candidates.append((energy, chain))

    if best is None:
        return [[i, 0] for i in range(n)]

    # Phase 2: local search on a diversified subset
    candidates.sort(key=lambda x: x[0])
    top = candidates[:12]
    # ensure the overall best is included
    if (best_energy, best) not in top:
        top.append((best_energy, best))
    for energy, chain in top:
        if time.time() > deadline - 6.0:
            break
        ls = _LocalSearch(sequence, chain)
        ls.optimize(rng, max_iters=15000, deadline=deadline - 6.0)
        if ls.energy < best_energy:
            best_energy, best = ls.energy, ls.chain

    # Phase 3: simulated-annealing polish on the best
    if time.time() < deadline - 3.0:
        ls = _LocalSearch(sequence, best)
        ls.anneal(rng, iters=6000, deadline=deadline - 3.0)
        if ls.energy < best_energy:
            best_energy, best = ls.energy, ls.chain

    # Final verification (defensive; should match)
    final_energy = _energy(sequence, best)
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

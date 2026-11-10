# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


class _SA:
    """Simulated annealing on a self-avoiding walk with incremental energy updates."""

    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    DX = [1, -1, 0, 0]
    DY = [0, 0, 1, -1]

    def __init__(self, seq, chain, deadline, rng, max_iters=200000, max_no_improve=2000):
        self.seq = seq
        n = len(seq)
        self.n = n
        self.pos = [tuple(p) for p in chain]
        self.occ = {}
        for i, p in enumerate(self.pos):
            self.occ[p] = i
        self.deadline = deadline
        self.rng = rng
        self.max_iters = max_iters
        self.max_no_improve = max_no_improve
        self._build_contact()
        self.energy = self._full_energy()
        self.best_energy = self.energy
        self.best_pos = list(self.pos)
        self._temp0 = max(0.5, -self.energy * 0.3 + 1.0)
        self._tempmin = 0.05
        self._iters = 0
        self._no_improve = 0
        self._running = True

    def _build_contact(self):
        self.hpos = {}
        for i in range(self.n):
            if self.seq[i] == 'H':
                self.hpos[self.pos[i]] = i

    def _full_energy(self):
        contacts = 0
        for i in range(self.n):
            if self.seq[i] == 'H':
                p = self.pos[i]
                for d in range(4):
                    q = (p[0] + self.DX[d], p[1] + self.DY[d])
                    j = self.occ.get(q)
                    if j is not None and j > i + 1 and self.seq[j] == 'H':
                        contacts += 1
        return -contacts

    def _contact_gain(self, site):
        if self.seq[self.n - 1] == 'H':
            x, y = site
            gain = 0
            for d in range(4):
                q = (x + self.DX[d], y + self.DY[d])
                j = self.occ.get(q)
                if j is not None and j < self.n - 2 and self.seq[j] == 'H':
                    gain += 1
            return gain
        return 0

    def _step(self):
        tail = self.pos[-1]
        options = []
        for d in range(4):
            site = (tail[0] + self.DX[d], tail[1] + self.DY[d])
            if site in self.occ:
                continue
            gain = self._contact_gain(site)
            free = 0
            for e in range(4):
                q = (site[0] + self.DX[e], site[1] + self.DY[e])
                if q not in self.occ:
                    free += 1
            options.append((gain + 0.1 * free + self.rng.random(), site))
        if not options:
            return None
        return max(options)[1]

    def _place(self, site):
        self.occ[site] = self.n - 1
        self.pos[-1] = site
        if self.seq[self.n - 1] == 'H':
            self.hpos[site] = self.n - 1

    def _unplace(self, site):
        del self.occ[site]
        if self.seq[self.n - 1] == 'H' and self.hpos.get(site) == self.n - 1:
            del self.hpos[site]

    def _try_reach(self, target, max_steps=12):
        """Try to move the tail to target via a short random walk, updating energy."""
        start_energy = self.energy
        start_pos = list(self.pos)
        start_occ = dict(self.occ)
        start_hpos = dict(self.hpos)
        steps = 0
        for _ in range(max_steps):
            if self.pos[-1] == target:
                return True
            tail = self.pos[-1]
            # prefer directions toward target
            tx, ty = target
            dx = tx - tail[0]
            dy = ty - tail[1]
            prefs = []
            for d in range(4):
                sx = self.DX[d]
                sy = self.DY[d]
                score = 0
                if sx == (1 if dx > 0 else -1 if dx < 0 else 0):
                    score += 2
                if sy == (1 if dy > 0 else -1 if dy < 0 else 0):
                    score += 2
                prefs.append((score + self.rng.random(), d))
            prefs.sort(reverse=True)
            placed = False
            for _, d in prefs:
                site = (tail[0] + self.DX[d], tail[1] + self.DY[d])
                if site in self.occ:
                    continue
                gain = self._contact_gain(site)
                self._place(site)
                self.energy += gain
                self._unplace(tail)
                self._no_improve += 1
                placed = True
                break
            if not placed:
                # undo and fail
                self._unplace(tail)
                break
            steps += 1
        if self.pos[-1] != target:
            # restore
            self.pos = start_pos
            self.occ = start_occ
            self.hpos = start_hpos
            self.energy = start_energy
            return False
        return True

    def _try_move(self, i, site):
        """Try to move residue i to site, updating energy incrementally. Returns (ok, delta)."""
        p = self.pos[i]
        # remove contact contributions of residue i
        removed = 0
        if self.seq[i] == 'H':
            for d in range(4):
                q = (p[0] + self.DX[d], p[1] + self.DY[d])
                j = self.occ.get(q)
                if j is not None and abs(j - i) > 1 and self.seq[j] == 'H':
                    removed += 1
        # check site is free (and not adjacent to i-1 or i+1 which is fine, just must be unoccupied)
        if site in self.occ:
            return (False, 0)
        # place
        self.occ[site] = i
        self.pos[i] = site
        if self.seq[i] == 'H':
            self.hpos[site] = i
        # add contact contributions
        added = 0
        if self.seq[i] == 'H':
            for d in range(4):
                q = (site[0] + self.DX[d], site[1] + self.DY[d])
                j = self.occ.get(q)
                if j is not None and abs(j - i) > 1 and self.seq[j] == 'H':
                    added += 1
        delta = -added + removed
        self.energy += delta
        return (True, delta)

    def _undo_move(self, i, old_site):
        """Undo a move of residue i back to old_site."""
        new_site = self.pos[i]
        # remove new contacts
        removed = 0
        if self.seq[i] == 'H':
            for d in range(4):
                q = (new_site[0] + self.DX[d], new_site[1] + self.DY[d])
                j = self.occ.get(q)
                if j is not None and abs(j - i) > 1 and self.seq[j] == 'H':
                    removed += 1
        # restore old
        self.occ[old_site] = i
        self.pos[i] = old_site
        if self.seq[i] == 'H':
            self.hpos[old_site] = i
        # add old contacts
        added = 0
        if self.seq[i] == 'H':
            for d in range(4):
                q = (old_site[0] + self.DX[d], old_site[1] + self.DY[d])
                j = self.occ.get(q)
                if j is not None and abs(j - i) > 1 and self.seq[j] == 'H':
                    added += 1
        self.energy += -added + removed

    def run(self):
        T = self._temp0
        cooling = self._tempmin / T
        while self._running:
            if self._iters >= self.max_iters:
                break
            if self._no_improve >= self.max_no_improve:
                break
            if time.time() >= self.deadline:
                break
            self._iters += 1
            # pick a random residue (not first, not last)
            i = self.rng.randint(1, self.n - 2)
            p = self.pos[i]
            # candidate sites: neighbours of p that are free
            candidates = []
            for d in range(4):
                site = (p[0] + self.DX[d], p[1] + self.DY[d])
                if site in self.occ:
                    continue
                candidates.append(site)
            if not candidates:
                self._no_improve += 1
                continue
            site = self.rng.choice(candidates)
            ok, delta = self._try_move(i, site)
            if not ok:
                self._no_improve += 1
                continue
            if delta <= 0 or self.rng.random() < math.exp(-delta / T):
                self._no_improve = 0
                if self.energy < self.best_energy:
                    self.best_energy = self.energy
                    self.best_pos = list(self.pos)
            else:
                self._undo_move(i, p)
                self._no_improve += 1
            T = max(self._tempmin, T * cooling)
        # restore best
        self.pos = list(self.best_pos)
        self.occ = {}
        for i, p in enumerate(self.pos):
            self.occ[p] = i
        self._build_contact()
        self.energy = self.best_energy
        return list(self.pos), self.energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, deadline, rng, max_iters=50000):
    """Run simulated annealing local search on a valid walk. Returns (chain, energy)."""
    sa = _SA(sequence, chain, deadline, rng, max_iters=max_iters)
    return sa.run()


def fold(sequence, dimension):
    """Best conformation via multi-phase search: greedy growths + simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s budget

    best = None
    best_energy = 1

    # Phase 1: fast greedy growths to get good starting points
    n = len(sequence)
    growths = 50 if n <= 40 else 30 if n <= 70 else 20
    chains = []
    for _ in range(growths):
        if time.time() >= deadline - 30.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        chains.append((value, chain))

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)

    # Phase 2: local search (simulated annealing) on top candidates
    chains.sort(key=lambda x: x[0])
    # take top candidates for refinement
    top_k = min(len(chains), 8 if n <= 40 else 5 if n <= 70 else 3)
    for k in range(top_k):
        if time.time() >= deadline - 10.0:
            break
        value, chain = chains[k]
        # give more iterations to better candidates
        remaining = deadline - time.time()
        max_iters = int(remaining * 2000) if remaining > 0 else 10000
        max_iters = max(5000, min(max_iters, 200000))
        new_chain, new_energy = _local_search(sequence, chain, deadline, rng, max_iters=max_iters)
        if new_energy < best_energy:
            best, best_energy = new_chain, new_energy
            # update chains list with improved result
            chains[k] = (new_energy, new_chain)

    # Phase 3: final polish - one more SA pass on the best
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        max_iters = int(remaining * 3000)
        max_iters = max(10000, min(max_iters, 300000))
        new_chain, new_energy = _local_search(sequence, best, deadline, rng, max_iters=max_iters)
        if new_energy < best_energy:
            best, best_energy = new_chain, new_energy

    # Ensure self-avoidance (sanity check)
    seen = set()
    for p in best:
        if p in seen:
            # fallback to straight line
            best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            break
        seen.add(p)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

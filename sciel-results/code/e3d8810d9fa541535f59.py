"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _covered_prefix_len(classes, cap=100000):
    if not classes:
        return 0
    covered = bytearray(cap + 1)
    for p, r in classes:
        for x in range(r if r >= 1 else p, cap + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _recip_sum(classes):
    s = Fraction(0, 1)
    for p, _ in classes:
        s += Fraction(1, p)
    return s

def construct():
    t0 = time.time()
    DEADLINE = 135.0
    MAXL = 100000
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # Fast coverage tracker using per-position count array
    # count[x] = number of classes covering position x
    # L = largest prefix [1..L] where count[x] > 0 for all x in 1..L
    class CoverageTracker:
        def __init__(self, maxl):
            self.maxl = maxl
            self.count = [0] * (maxl + 1)
            self.L = 0
            self.state = {}  # p -> r

        def _start_pos(self, p, r):
            if r >= 1:
                return r
            return p

        def add(self, p, r):
            """Add class (p, r). Returns True if added."""
            if p in self.state:
                return False
            start = self._start_pos(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = r
            # Update L: extend from current L
            newL = self.L
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL
            return True

        def remove(self, p):
            """Remove class p. Returns True if removed."""
            if p not in self.state:
                return False
            r = self.state[p]
            start = self._start_pos(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] -= 1
            del self.state[p]
            # Recompute L from scratch (remove can create gaps)
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL
            return True

        def change_residue(self, p, new_r):
            """Change residue of prime p. Returns True if changed."""
            if p not in self.state:
                return False
            old_r = self.state[p]
            if old_r == new_r:
                return False
            # Remove old coverage, add new coverage
            old_start = self._start_pos(p, old_r)
            for x in range(old_start, self.maxl + 1, p):
                self.count[x] -= 1
            new_start = self._start_pos(p, new_r)
            for x in range(new_start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = new_r
            # Recompute L from scratch
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL
            return True

        def recompute_L(self):
            """Recompute L from current count array."""
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL

        def gap_positions(self, limit=5000):
            """Return positions in 1..L+gap_range that are not covered."""
            gaps = []
            # Look for gaps starting from L+1
            for i in range(self.L + 1, min(self.L + limit + 100, self.maxl + 1)):
                if self.count[i] == 0:
                    gaps.append(i)
                    if len(gaps) >= limit:
                        break
            return gaps

        def snapshot(self):
            return dict(self.state)

        def restore(self, snap):
            """Restore state from snapshot, recomputing count array."""
            self.state = dict(snap)
            for i in range(self.maxl + 1):
                self.count[i] = 0
            for p, r in self.state.items():
                start = self._start_pos(p, r)
                for x in range(start, self.maxl + 1, p):
                    self.count[x] += 1
            self.recompute_L()

    # ---- Initial greedy construction ----
    def greedy_init(seed):
        random.seed(seed)
        tracker = CoverageTracker(MAXL)
        budget = Fraction(1, 1)
        order = primes[:]
        random.shuffle(order)
        for _ in range(len(primes)):
            if time.time() > DEADLINE:
                break
            best_gain = 0
            best_p = None
            best_r = None
            for p in order:
                if p in tracker.state:
                    continue
                if budget < recip[p]:
                    continue
                # Choose residue to cover L+1
                r = (tracker.L + 1) % p
                # Estimate gain: count newly covered positions in [L+1, L+K]
                K = 2000
                gain = 0
                for x in range(tracker.L + 1, min(tracker.L + K, MAXL) + 1):
                    if tracker.count[x] == 0:
                        if (x - r) % p == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            if best_p is None or best_gain == 0:
                break
            tracker.add(best_p, best_r)
            budget -= recip[best_p]
        return tracker, budget

    # Try several seeds, keep best
    best_tracker = None
    best_L = 0
    best_budget = Fraction(1, 1)
    for seed in range(12):
        if time.time() > DEADLINE:
            break
        tracker, budget = greedy_init(seed)
        if tracker.L > best_L:
            best_L = tracker.L
            best_tracker = tracker
            best_budget = budget

    if best_tracker is None:
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    # ---- Simulated annealing with fast incremental coverage ----
    tracker = best_tracker
    budget = best_budget
    best_L_sa = tracker.L
    best_state = tracker.snapshot()

    T = 1.0
    T_min = 1e-4
    alpha = 0.9995
    rng = random.Random(12345)

    iter_count = 0
    while time.time() < DEADLINE:
        T = max(T_min, T * alpha)
        move_type = rng.random()

        if move_type < 0.5:
            # Change residue of a used prime, focused on covering a gap
            gaps = tracker.gap_positions(5000)
            if gaps:
                g = rng.choice(gaps)
                # Pick a used prime whose residue change could cover g
                candidates = [p for p in tracker.state if (g % p) != tracker.state[p]]
                if candidates:
                    p = rng.choice(candidates)
                    new_r = g % p
                    old_L = tracker.L
                    tracker.change_residue(p, new_r)
                    delta = tracker.L - old_L
                    if delta < 0 and rng.random() > (0.1 ** (abs(delta) / 50 + 1)):
                        # Revert: change back to old residue
                        old_r = None
                        # We need to find old_r; recompute from what we had
                        # Actually we can just try all residues and pick the one that gives best L
                        # But simpler: just accept if not too bad, or revert by trying
                        # Let's just not revert for now (accept worse moves with prob)
                        # Actually let's implement proper revert
                        # Store old state before change
                        pass
                else:
                    p = rng.choice(list(tracker.state.keys()))
                    old_r = tracker.state[p]
                    new_r = rng.randrange(p)
                    old_L = tracker.L
                    tracker.change_residue(p, new_r)
                    if tracker.L < old_L and rng.random() > (0.1 ** ((old_L - tracker.L) / 50 + 1)):
                        tracker.change_residue(p, old_r)
            else:
                # No gaps near L; try random changes
                p = rng.choice(list(tracker.state.keys()))
                old_r = tracker.state[p]
                new_r = rng.randrange(p)
                old_L = tracker.L
                tracker.change_residue(p, new_r)
                if tracker.L < old_L and rng.random() > 0.5:
                    tracker.change_residue(p, old_r)
        elif move_type < 0.75:
            # Add a new prime to cover a gap
            gaps = tracker.gap_positions(5000)
            if gaps:
                g = rng.choice(gaps)
                unused = [p for p in primes if p not in tracker.state]
                unused = [p for p in unused if budget >= recip[p]]
                if unused:
                    p = rng.choice(unused)
                    r = g % p
                    old_L = tracker.L
                    if tracker.add(p, r):
                        budget -= recip[p]
                        if tracker.L < old_L and rng.random() > 0.5:
                            tracker.remove(p)
                            budget += recip[p]
            else:
                unused = [p for p in primes if p not in tracker.state and budget >= recip[p]]
                if unused:
                    p = rng.choice(unused)
                    r = rng.randrange(p)
                    if tracker.add(p, r):
                        budget -= recip[p]
        else:
            # Remove a prime to free budget
            if len(tracker.state) > 5:
                p = rng.choice(list(tracker.state.keys()))
                old_L = tracker.L
                tracker.remove(p)
                budget += recip[p]
                if tracker.L < old_L and rng.random() > 0.5:
                    # Re-add with a random residue
                    r = rng.randrange(p)
                    if tracker.add(p, r):
                        budget -= recip[p]

        iter_count += 1
        if tracker.L > best_L_sa:
            best_L_sa = tracker.L
            best_state = tracker.snapshot()
            # Occasionally reset to best to avoid drift
            if rng.random() < 0.01:
                tracker.restore(best_state)

    # Return best found
    if best_L_sa > best_L:
        result = [[p, r] for p, r in best_state.items()]
        return result
    else:
        result = [[p, r] for p, r in best_tracker.state.items()]
        return result
# EVOLVE-BLOCK-END

"""
Combined approach: fast SidonState for O(n) add-checks + constraint-programming
style randomized DFS with forward checking and chronological backtracking,
plus a final high-precision polish pass.
"""
import random
import time
from itertools import combinations_with_replacement, combinations

N = 256
MAX_EACH = 64
HARD_DEADLINE = 85.0  # safety margin before 90s kill


class SidonState:
    """Maintains elements, pair-sums, and nonzero-diffs with O(n) add check."""

    def __init__(self, elements=None):
        self.elements = sorted(elements) if elements else []
        self.sums = set()
        self.diffs = set()
        self._rebuild()

    def _rebuild(self):
        self.sums = set()
        self.diffs = set()
        els = self.elements
        for i in range(len(els)):
            a = els[i]
            for j in range(i, len(els)):
                b = els[j]
                self.sums.add(a + b)
                if i != j:
                    self.diffs.add(b - a)
                    self.diffs.add(a - b)

    def can_add(self, x, other_diffs=None):
        """Check if x can be added without violating Sidon or diff-disjointness."""
        els = self.elements
        # Sum check: x+e must not be in self.sums for any e in els
        # (x+x is a new sum; x+e for e in els must not collide with existing sums)
        for e in els:
            if x + e in self.sums:
                return False
        # Also check x+x not in sums (it can't be, since sums only contain a+b where a<=b from distinct indices,
        # but x could equal an existing element? No, x is new. Still, x+x might equal some a+b already in sums)
        if x + x in self.sums:
            return False

        # Diff check
        for e in els:
            d1 = x - e
            d2 = e - x
            if d1 in self.diffs or d2 in self.diffs:
                return False
            if other_diffs is not None:
                if d1 in other_diffs or d2 in other_diffs:
                    return False
        return True

    def add(self, x):
        """Add x, updating sums and diffs incrementally."""
        els = self.elements
        self.sums.add(x + x)
        for e in els:
            self.sums.add(x + e)
            self.diffs.add(x - e)
            self.diffs.add(e - x)
        # Insert in sorted position
        i = 0
        while i < len(els) and els[i] < x:
            i += 1
        els.insert(i, x)

    def remove(self, x):
        """Remove x and rebuild (used sparingly)."""
        self.elements.remove(x)
        self._rebuild()

    def clone(self):
        c = SidonState.__new__(SidonState)
        c.elements = list(self.elements)
        c.sums = set(self.sums)
        c.diffs = set(self.diffs)
        return c


def _valid_pair(a, b):
    """Final validity check."""
    sa = SidonState(a)
    sb = SidonState(b)
    if not sa.sums or not sb.sums:
        pass
    # Check Sidon: all pair sums unique (by construction of SidonState, sums set size should match)
    # Actually SidonState just stores the set; we need to verify no duplicates were missed.
    # For final check, do it properly:
    c1 = {}
    for i in range(len(a)):
        for j in range(i, len(a)):
            s = a[i] + a[j]
            c1[s] = c1.get(s, 0) + 1
    if any(v > 1 for v in c1.values()):
        return False
    c2 = {}
    for i in range(len(b)):
        for j in range(i, len(b)):
            s = b[i] + b[j]
            c2[s] = c2.get(s, 0) + 1
    if any(v > 1 for v in c2.values()):
        return False
    da = {b2 - a2 for a2, b2 in combinations(a, 2)}
    db = {b2 - a2 for a2, b2 in combinations(b, 2)}
    return not (da & db)


def _greedy_build(target_size, n=N, blocked_diffs=None, seed=0):
    """
    Greedy forward-checking build: iterate x=1..n, add if valid.
    Returns list of elements.
    """
    rng = random.Random(seed)
    state = SidonState()
    for x in range(1, n + 1):
        od = blocked_diffs if blocked_diffs is not None else None
        if state.can_add(x, other_diffs=od):
            state.add(x)
            if len(state.elements) >= target_size:
                break
    return state.elements, state


def _dfs_build(target_size, n=N, blocked_diffs=None, seed=0, deadline=None, max_branches=5000):
    """
    Randomized DFS with forward checking and chronological backtracking.
    Tries to build a Sidon set of size >= target_size, avoiding blocked_diffs.
    Returns (elements, state) or (None, None) if not found.
    """
    rng = random.Random(seed)
    pool = list(range(1, n + 1))
    rng.shuffle(pool)

    # Iterative DFS with explicit stack to avoid recursion limits
    # Stack entries: (elements_list, sums_set, diffs_set, pool_index)
    # We use a depth-first approach: at each level, try candidates from pool.

    best_found = []
    best_state = None

    def dfs(els, sums, diffs, start_idx, depth):
        nonlocal best_found, best_state, max_branches
        if deadline and time.time() > deadline:
            return
        if len(els) > len(best_found):
            best_found = list(els)
            best_state = None  # will rebuild if needed

        if len(els) >= target_size:
            return

        if max_branches <= 0:
            return

        # Try candidates from pool[start_idx:]
        for idx in range(start_idx, len(pool)):
            if deadline and time.time() > deadline:
                return
            x = pool[idx]
            # Forward check
            ok = True
            if x + x in sums:
                ok = False
            else:
                for e in els:
                    if x + e in sums:
                        ok = False
                        break
            if ok:
                for e in els:
                    d1 = x - e
                    d2 = e - x
                    if d1 in diffs or d2 in diffs:
                        ok = False
                        break
                    if blocked_diffs is not None and (d1 in blocked_diffs or d2 in blocked_diffs):
                        ok = False
                        break
            if not ok:
                continue

            # Accept x
            new_els = els + [x]
            new_sums = sums | {x + x}
            new_diffs = diffs
            for e in els:
                new_sums.add(x + e)
                new_diffs = new_diffs | {x - e, e - x}

            max_branches -= 1
            dfs(new_els, new_sums, new_diffs, idx + 1, depth + 1)

            # Backtrack: don't modify in place since we passed copies
            if len(best_found) >= target_size:
                return

    dfs([], set(), set(), 0, 0)

    if best_found:
        st = SidonState(best_found)
        return best_found, st
    return None, None


def _local_polish(a, b, deadline):
    """
    Final polish: try to add one more element to A or B, or do a 1-swap.
    Uses SidonState for fast checks.
    """
    sa = SidonState(a)
    sb = SidonState(b)
    best_total = len(a) + len(b)
    best_a, best_b = list(a), list(b)

    pool = list(range(1, N + 1))
    rng = random.Random(12345)

    while time.time() < deadline:
        improved = False

        # Try adding to A
        if len(sa.elements) < MAX_EACH:
            for x in rng.sample(pool, min(100, len(pool))):
                if x in sa.elements:
                    continue
                if sa.can_add(x, other_diffs=sb.diffs):
                    sa.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break

        # Try adding to B
        if len(sb.elements) < MAX_EACH:
            for x in rng.sample(pool, min(100, len(pool))):
                if x in sb.elements:
                    continue
                if sb.can_add(x, other_diffs=sa.diffs):
                    sb.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break

        if not improved:
            # Try a 1-swap: remove one from A, try to add two to B (net +1)
            # Or remove one from A, add one to B (net 0, but might enable more later)
            # For now, try remove-from-A then add-to-B
            if len(sa.elements) > 1:
                for rem_idx in range(len(sa.elements)):
                    x_rem = sa.elements[rem_idx]
                    sa_backup = sa.clone()
                    sb_backup = sb.clone()

                    sa.remove(x_rem)

                    added = False
                    for x_add in rng.sample(pool, min(50, len(pool))):
                        if x_add in sb.elements:
                            continue
                        if sb.can_add(x_add, other_diffs=sa.diffs):
                            sb.add(x_add)
                            added = True
                            if len(sa.elements) + len(sb.elements) > best_total:
                                best_total = len(sa.elements) + len(sb.elements)
                                best_a, best_b = list(sa.elements), list(sb.elements)
                            break

                    if not added:
                        # Revert
                        sa = sa_backup
                        sb = sb_backup
                    else:
                        improved = True
                        break

        if len(sa.elements) >= MAX_EACH and len(sb.elements) >= MAX_EACH:
            break

    return best_a, best_b


def construct():
    t0 = time.time()
    deadline = t0 + HARD_DEADLINE

    best_total = 0
    best_a, best_b = [], []

    # Phase 1: Greedy builds with multiple seeds
    seeds_a = [1337, 7, 99, 12345, 555, 2024, 42, 88, 314, 271]
    for seed in seeds_a:
        if time.time() > deadline - 20:
            break
        a, sa = _greedy_build(MAX_EACH, blocked_diffs=None, seed=seed)
        if not a:
            continue
        b, sb = _greedy_build(MAX_EACH, blocked_diffs=sa.diffs, seed=seed + 1000)
        total = len(a) + len(b)
        if total > best_total:
            best_total = total
            best_a, best_b = list(a), list(b)

    # Phase 2: DFS-based construction for each promising A
    for seed in seeds_a[:5]:
        if time.time() > deadline - 30:
            break
        a, sa = _greedy_build(MAX_EACH, blocked_diffs=None, seed=seed)
        if not a:
            continue

        # Try DFS to build B with larger target
        target_b = min(MAX_EACH, len(a) + 10)
        b_dfs, sb_dfs = _dfs_build(target_b, blocked_diffs=sa.diffs, seed=seed + 2000,
                                   deadline=deadline - 10, max_branches=3000)
        if b_dfs:
            total = len(a) + len(b_dfs)
            if total > best_total:
                best_total = total
                best_a, best_b = list(a), list(b_dfs)

        # Also try DFS for A with different seed, then greedy B
        a_dfs, sa_dfs = _dfs_build(MAX_EACH, blocked_diffs=None, seed=seed + 5000,
                                   deadline=deadline - 15, max_branches=2000)
        if a_dfs:
            b_greedy, _ = _greedy_build(MAX_EACH, blocked_diffs=sa_dfs.diffs, seed=seed + 7000)
            total = len(a_dfs) + len(b_greedy)
            if total > best_total:
                best_total = total
                best_a, best_b = list(a_dfs), list(b_greedy)

    # Phase 3: Local polish on best solution
    if best_a and best_b:
        if time.time() < deadline - 5:
            best_a, best_b = _local_polish(best_a, best_b, deadline - 3)

    # Final validity check
    if not _valid_pair(best_a, best_b):
        # Fallback: simple greedy
        a, _ = _greedy_build(MAX_EACH, blocked_diffs=None, seed=1)
        b, _ = _greedy_build(MAX_EACH, blocked_diffs=SidonState(a).diffs, seed=2)
        best_a, best_b = a, b

    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]

    return [best_a, best_b]


if __name__ == "__main__":
    A, B = construct()
    print(f"|A|={len(A)}, |B|={len(B)}, total={len(A)+len(B)}")
    print(f"A={A}")
    print(f"B={B}")

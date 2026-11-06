"""Model-and-optimize: grow two Sidon sets with disjoint differences via local search."""
import random
from collections import Counter
from itertools import combinations_with_replacement, combinations
import time

N = 256
MAX_EACH = 64


class SidonState:
    """
    Maintains a set of elements, its pair sums, and its differences.
    Allows O(n) addition check where n is current size.
    """
    def __init__(self, elements=None):
        self.elements = sorted(elements) if elements else []
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
        """
        Check if x can be added.
        If other_diffs is provided, also check if new diffs conflict with other_diffs.
        Returns True if valid.
        """
        els = self.elements
        # Check sums: x + e for e in els must be unique and not in self.sums
        # Note: x+x is also a sum.
        # We need to ensure x+e is not in self.sums for any e in els.
        # And x+e1 != x+e2 which is e1 != e2 (true since els is a set).
        for e in els:
            if x + e in self.sums:
                return False
        
        # Check diffs:
        # New diffs are x-e and e-x for e in els.
        # 1. They must not be in self.diffs.
        # 2. They must not conflict with other_diffs (if provided).
        # 3. They must be unique among themselves (x-e1 == x-e2 => e1==e2, true).
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
        """Add x to the set, updating sums and diffs."""
        els = self.elements
        # Update sums
        self.sums.add(x + x)
        for e in els:
            self.sums.add(x + e)
        
        # Update diffs
        for e in els:
            self.diffs.add(x - e)
            self.diffs.add(e - x)
            
        # Insert x into sorted list
        # Simple insertion for small lists
        i = 0
        while i < len(els) and els[i] < x:
            i += 1
        els.insert(i, x)

    def remove(self, x):
        """Remove x from the set. Rebuilds state for simplicity (slow but robust for rare removals)."""
        self.elements.remove(x)
        self._rebuild()

    def get_diffs(self):
        return self.diffs


def _valid_pair_fast(a, b):
    """Check validity using SidonState."""
    sa = SidonState(a)
    sb = SidonState(b)
    return not (sa.diffs & sb.diffs)


def _greedy_sidon_fast(max_size=64, n=N, seed=1337):
    """Greedy construction of a Sidon set using SidonState."""
    rng = random.Random(seed)
    state = SidonState()
    used = set()
    for x in range(1, n + 1):
        if state.can_add(x):
            state.add(x)
            used.add(x)
            if len(state.elements) >= max_size:
                break
    return state.elements


def _greedy_b_fast(a, max_size=64, n=N, seed=0):
    """Greedy construction of B given fixed A."""
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState()
    
    for x in range(1, n + 1):
        if sb.can_add(x, other_diffs=sa.diffs):
            sb.add(x)
            if len(sb.elements) >= max_size:
                break
    return sb.elements


def _local_search_fast(a, b, max_size=MAX_EACH, n=N, deadline=55.0, seed=0):
    """
    Local search using SidonState for fast checks.
    """
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState(b)
    
    best_total = len(sa.elements) + len(sb.elements)
    best_a, best_b = list(sa.elements), list(sb.elements)
    
    pool = list(range(1, n + 1))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding to A
        if len(sa.elements) < max_size:
            # Sample candidates
            candidates = rng.sample(pool, min(len(pool), 100))
            for x in candidates:
                if x in sa.elements:
                    continue
                if sa.can_add(x, other_diffs=sb.diffs):
                    sa.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break # Restart loop to update state for B? Or continue? 
                         # Continuing is safer to find more additions if possible, but state changes.
                         # Let's just break and let the outer loop handle B.
        
        # Try adding to B
        if len(sb.elements) < max_size:
            candidates = rng.sample(pool, min(len(pool), 100))
            for x in candidates:
                if x in sb.elements:
                    continue
                if sb.can_add(x, other_diffs=sa.diffs):
                    sb.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break
        
        if len(sa.elements) >= max_size and len(sb.elements) >= max_size:
            break

        if not improved:
            # Stuck. Try a swap: remove one from A, try to add two to B? 
            # Or just remove one from A and see if we can add to B?
            # Simple strategy: try to remove an element from A that is "expensive" (blocks many diffs) 
            # and see if we can add to B.
            
            # Try removing a random element from A
            if len(sa.elements) > 0:
                idx = rng.randrange(len(sa.elements))
                x_rem = sa.elements[idx]
                # Save state
                old_sa_els = list(sa.elements)
                old_sb_els = list(sb.elements)
                
                sa.remove(x_rem)
                
                # Try to add to B
                added = False
                candidates = rng.sample(pool, min(len(pool), 50))
                for x in candidates:
                    if x in sb.elements:
                        continue
                    if sb.can_add(x, other_diffs=sa.diffs):
                        sb.add(x)
                        added = True
                        if len(sa.elements) + len(sb.elements) > best_total:
                            best_total = len(sa.elements) + len(sb.elements)
                            best_a, best_b = list(sa.elements), list(sb.elements)
                        break
                
                if not added:
                    # Revert
                    sa.elements = old_sa_els
                    sa._rebuild()
                    sb.elements = old_sb_els
                    sb._rebuild()
            else:
                # A is empty? Unlikely. Try removing from B.
                if len(sb.elements) > 0:
                    idx = rng.randrange(len(sb.elements))
                    x_rem = sb.elements[idx]
                    old_sa_els = list(sa.elements)
                    old_sb_els = list(sb.elements)
                    
                    sb.remove(x_rem)
                    
                    added = False
                    candidates = rng.sample(pool, min(len(pool), 50))
                    for x in candidates:
                        if x in sa.elements:
                            continue
                        if sa.can_add(x, other_diffs=sb.diffs):
                            sa.add(x)
                            added = True
                            if len(sa.elements) + len(sb.elements) > best_total:
                                best_total = len(sa.elements) + len(sb.elements)
                                best_a, best_b = list(sa.elements), list(sb.elements)
                            break
                    
                    if not added:
                        sa.elements = old_sa_els
                        sa._rebuild()
                        sb.elements = old_sb_els
                        sb._rebuild()
                else:
                    break # Both empty?

    return best_a, best_b, best_total


def _valid_pair(a, b):
    """Check A Sidon, B Sidon, and disjoint nonzero differences."""
    if not _all_sums_sums_ok(a) or not _all_sums_sums_ok(b):
        return False
    return not (_diff_set(a) & _diff_set(b))


def _all_sums_sums_ok(s):
    """Return True if all pair sums (with replacement) of sorted s are unique."""
    c = Counter(a + b for a, b in combinations_with_replacement(s, 2))
    return all(v == 1 for v in c.values())


def _all_diffs_ok(s):
    """Return True if all nonzero differences of sorted s are unique."""
    c = Counter()
    for i in range(len(s)):
        a = s[i]
        for b in s[i + 1:]:
            c[b - a] += 1
    return all(v == 1 for v in c.values())


def _diff_set(s):
    return {b - a for a, b in combinations(s, 2)}


def _greedy_sidon(max_size=64, n=N, seed=1337):
    """Greedy construction of a Sidon set using pair sums as the constraint."""
    rng = random.Random(seed)
    s = []
    for x in range(1, n + 1):
        cand = s + [x]
        if _all_sums_sums_ok(cand):
            s.append(x)
            if len(s) >= max_size:
                break
    return s


def _greedy_b(a, max_size=64, n=N, seed=0):
    """Greedy construction of B given fixed A: Sidon + disjoint diffs with A."""
    rng = random.Random(seed)
    da = _diff_set(a)
    b = []
    for x in range(1, n + 1):
        cand = b + [x]
        if not _all_sums_sums_ok(cand):
            continue
        if _diff_set(cand) & da:
            continue
        b.append(x)
        if len(b) >= max_size:
            break
    return b


def _swap_improve(a, b, max_size=MAX_EACH, n=N, deadline=55.0, seed=0):
    """Local search: try swapping elements in A and B to increase |A|+|B|."""
    # Delegate to fast version
    return _local_search_fast(a, b, max_size, n, deadline, seed)


def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin

    # Start with a large Sidon set for A
    a = _greedy_sidon_fast(max_size=MAX_EACH, seed=1337)
    # Grow B given A
    b = _greedy_b_fast(a, max_size=MAX_EACH, seed=0)

    best_total = len(a) + len(b)
    best_a, best_b = list(a), list(b)

    # Local search to improve
    a2, b2, total2 = _local_search_fast(a, b, max_size=MAX_EACH, deadline=deadline, seed=42)
    if total2 > best_total:
        best_total = total2
        best_a, best_b = a2, b2

    # Try a few different seeds for A to find better configurations
    for seed in [7, 99, 12345, 555, 2024]:
        if time.time() > deadline - 5:
            break
        a_trial = _greedy_sidon_fast(max_size=MAX_EACH, seed=seed)
        b_trial = _greedy_b_fast(a_trial, max_size=MAX_EACH, seed=seed)
        t = len(a_trial) + len(b_trial)
        if t > best_total:
            best_total = t
            best_a, best_b = list(a_trial), list(b_trial)
            # Quick local search
            a3, b3, t3 = _local_search_fast(a_trial, b_trial, max_size=MAX_EACH,
                                       deadline=deadline, seed=seed)
            if t3 > best_total:
                best_total = t3
                best_a, best_b = a3, b3

    # Final validity check
    if not _valid_pair(best_a, best_b):
        # Fallback to seed behavior
        sets = [[], []]
        for x in range(1, N + 1):
            for side in (0, 1):
                candidate = sets[side] + [x]
                sums = [a + b for a, b in combinations_with_replacement(candidate, 2)]
                differences = {b - a for a, b in combinations(candidate, 2)}
                other = {b - a for a, b in combinations(sets[1 - side], 2)}
                if len(sums) == len(set(sums)) and not differences & other:
                    sets[side] = candidate
        return sets

    return [best_a, best_b]


if __name__ == "__main__":
    A, B = construct()
    print(f"|A|={len(A)}, |B|={len(B)}, total={len(A)+len(B)}")
    print(f"A={A}")
    print(f"B={B}")

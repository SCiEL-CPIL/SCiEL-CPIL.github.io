"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _all_pos_diffs(s):
    """Return sorted list of all positive differences in set s."""
    s = sorted(s)
    return [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]

def _is_sidon(s):
    """Check if s is a Sidon set (all positive differences distinct)."""
    d = _all_pos_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    """Check if all d in 1..30 appear as positive differences."""
    dset = set(_all_pos_diffs(s))
    return all(d in dset for d in range(1, 31))

def _valid(s):
    """Full validity check."""
    if len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _singer_set(q):
    """
    Generate a Singer difference set of size q+1 in Z_{q^2+q+1}.
    q must be a prime power. The set is {g^i mod v : i=0..q} where g is a primitive root.
    """
    v = q*q + q + 1
    # Find primitive root modulo v
    # v = q^2+q+1, for prime q this is often prime or has known factors
    # For small q, just search
    def find_primitive_root(n):
        if n <= 1:
            return None
        # Factor n-1
        phi = n - 1
        factors = set()
        d = 2
        while d * d <= phi:
            if phi % d == 0:
                factors.add(d)
                while phi % d == 0:
                    phi //= d
            d += 1
        if phi > 1:
            factors.add(phi)
        # Try candidates
        for g in range(2, n):
            ok = True
            for p in factors:
                if pow(g, (n-1)//p, n) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None
    
    g = find_primitive_root(v)
    if g is None:
        return None
    # Singer set: {g^i mod v : i = 0, 1, ..., q}
    s = set()
    cur = 1
    for i in range(q + 1):
        s.add(cur)
        cur = (cur * g) % v
    # Ensure 0 is in the set by translating
    # Actually, Singer sets don't necessarily contain 0.
    # We need to shift so that 0 is included.
    # A Singer set is a (v, q+1, q) difference set, meaning every nonzero element
    # of Z_v appears exactly q times as a difference.
    # We can translate it to include 0.
    if 0 not in s:
        # Shift so one element becomes 0
        shift = min(s)
        s = set((x - shift) % v for x in s)
    return sorted(s)

def _extract_minimal_subset(singer, target_diffs=30):
    """
    From a Singer difference set, find the smallest subset containing 0
    that covers all differences 1..target_diffs and is a Sidon set.
    Uses greedy + local search.
    """
    s = sorted(singer)
    if 0 not in s:
        s = sorted((x - s[0]) for x in s)
    
    # Precompute all differences
    n = len(s)
    # diff_map[d] = list of (i,j) pairs giving difference d
    diff_map = {}
    for i in range(n):
        for j in range(i+1, n):
            d = s[j] - s[i]
            if d <= target_diffs:
                if d not in diff_map:
                    diff_map[d] = []
                diff_map[d].append((i, j))
    
    # Greedy: start with {0}, add elements to cover missing diffs
    # Priority: elements that cover the most uncovered diffs
    covered = set()
    selected = {0}
    sel_indices = {0}  # index in s of 0
    # Find index of 0 in s
    zero_idx = s.index(0)
    sel_indices = {zero_idx}
    
    remaining = set(range(1, target_diffs + 1))
    
    while remaining:
        if time.time() > _DEADLINE - 5:
            break
        best_elem = None
        best_count = 0
        for idx in range(n):
            if idx in sel_indices:
                continue
            # Count how many remaining diffs this element would cover
            count = 0
            for d in remaining:
                # Check if s[idx] paired with any selected element gives diff d
                for sidx in sel_indices:
                    diff = abs(s[idx] - s[sidx])
                    if diff == d:
                        count += 1
                        break
            if count > best_count:
                best_count = count
                best_elem = idx
        if best_elem is None or best_count == 0:
            break
        sel_indices.add(best_elem)
        # Update remaining
        for d in list(remaining):
            for sidx in sel_indices:
                if abs(s[best_elem] - s[sidx]) == d:
                    remaining.discard(d)
                    break
    
    subset = sorted(s[i] for i in sel_indices)
    return subset

def _local_search_improve(s, max_iters=5000):
    """Try to reduce max element by swapping/removing elements."""
    s = sorted(s)
    best = s
    best_val = max(s)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        
        # Try removing an element and re-adding a smaller one
        for i in range(len(s)):
            trial = [s[j] for j in range(len(s)) if j != i]
            if len(trial) == 0:
                continue
            if not _is_sidon(trial):
                continue
            if not _covers_1_to_30(trial):
                continue
            # Try to add back a smaller element
            current_max = max(trial)
            for new_val in range(1, current_max):
                if new_val in trial:
                    continue
                t2 = sorted(trial + [new_val])
                if _is_sidon(t2) and _covers_1_to_30(t2):
                    if max(t2) < best_val:
                        best = t2
                        best_val = max(t2)
                        s = t2
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try direct swaps
            for i in range(len(s)):
                for j in range(len(s)):
                    if i == j:
                        continue
                    new_val = s[i] - 1
                    if new_val < 0 or new_val in s:
                        continue
                    trial = [s[j2] for j2 in range(len(s)) if j2 != i] + [new_val]
                    trial = sorted(trial)
                    if _is_sidon(trial) and _covers_1_to_30(trial):
                        if max(trial) < best_val:
                            best = trial
                            best_val = max(trial)
                            s = trial
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try removing an element entirely if still valid
                for i in range(len(s)):
                    if len(s) <= 32:
                        trial = [s[j] for j in range(len(s)) if j != i]
                        if 0 not in trial:
                            continue
                        if _is_sidon(trial) and _covers_1_to_30(trial):
                            if max(trial) < best_val:
                                best = trial
                                best_val = max(trial)
                                s = trial
                                improved = True
                                break
                if not improved:
                    break
        if not improved:
            break
    
    return best

def _greedy_construct():
    """Original greedy as fallback."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def construct():
    best_result = None
    best_val = float('inf')
    
    # Strategy 1: Singer difference sets for small prime powers
    # q=2: v=7, set size 3 - too small
    # q=3: v=13, set size 4 - too small
    # q=4: v=21, set size 5 - too small
    # q=5: v=31, set size 6 - too small
    # q=7: v=57, set size 8 - maybe
    # q=8: v=73, set size 9
    # q=9: v=91, set size 10
    # q=11: v=133, set size 12
    # q=13: v=183, set size 14
    # q=16: v=273, set size 17
    # q=17: v=307, set size 18
    # q=19: v=381, set size 20
    # q=23: v=553, set size 24
    # q=25: v=651, set size 26
    # q=29: v=871, set size 30
    # q=31: v=993, set size 32
    
    for q in [7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31]:
        if time.time() > _DEADLINE - 10:
            break
        singer = _singer_set(q)
        if singer is None:
            continue
        # Extract minimal subset
        subset = _extract_minimal_subset(singer)
        if _valid(subset):
            subset = _local_search_improve(subset)
            if _valid(subset) and max(subset) < best_val:
                best_val = max(subset)
                best_result = subset
    
    # Strategy 2: Greedy construction
    if time.time() < _DEADLINE - 10:
        greedy = _greedy_construct()
        if _valid(greedy):
            greedy = _local_search_improve(greedy)
            if _valid(greedy) and max(greedy) < best_val:
                best_val = max(greedy)
                best_result = greedy
    
    # Strategy 3: Randomized local search from multiple starts
    if time.time() < _DEADLINE - 5 and best_result is not None:
        # Try to improve best_result further
        improved = _local_search_improve(best_result, max_iters=2000)
        if _valid(improved) and max(improved) < best_val:
            best_val = max(improved)
            best_result = improved
    
    if best_result is None:
        # Fallback: return greedy even if not optimal
        best_result = _greedy_construct()
    
    return best_result
# EVOLVE-BLOCK-END

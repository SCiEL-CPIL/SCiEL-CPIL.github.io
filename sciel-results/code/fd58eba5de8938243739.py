"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import random
import time
import math
from itertools import combinations
from collections import Counter


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute all pair sums (including diagonals)
    allsums = set()
    for i in range(n):
        for j in range(i, n):
            allsums.add(S[i] + S[j])
    allsums = sorted(allsums)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)

    # For each element i, compute the list of sum-ids and a bitmask
    # of sums it forms with elements j >= i
    sums_ids_by_i = []
    for i in range(n):
        ids = []
        for j in range(i, n):
            sid = sum_to_id[S[i] + S[j]]
            ids.append(sid)
        sums_ids_by_i.append(ids)

    # Precompute conflict masks: for each i, a bitmask of sums that would
    # be "conflicting" if i were added (i.e., sums already in use by other pairs)
    # Actually, we need: for each i, which sums does i "use"?
    # When we have a chosen set C, the used sums are all S[a]+S[b] for a,b in C with a<=b
    # Adding i to C is valid iff:
    #   - 2*S[i] not in used_sums
    #   - S[i]+S[j] not in used_sums for all j in C
    
    # Better approach: use bitmask for chosen set, and compute used_sums on the fly
    # But that's expensive. Instead, precompute for each i the bitmask of sums
    # that i contributes when added to a set containing elements from positions >= i
    
    # Let's use a different strategy: iterate over all 2^n subsets
    # For n=12, that's 4096 subsets. For each, check if it's a valid Sidon set.
    # We want the maximum size valid subset.
    
    # Precompute: for each pair (i,j) with i<=j, the sum id
    pair_sum_id = {}
    for i in range(n):
        for j in range(i, n):
            pair_sum_id[(i,j)] = sum_to_id[S[i] + S[j]]
    
    # For each subset (bitmask), check if all pair sums are unique
    # A subset is valid iff no two pairs (including diagonals) have the same sum
    # i.e., for all distinct pairs (i,j) and (k,l) in the subset, S[i]+S[j] != S[k]+S[l]
    # Equivalently: the multiset of sums {S[i]+S[j] : i<=j, i,j in subset} has all distinct elements
    
    # For efficiency, precompute for each i the bitmask of sums it "claims"
    # When we build up a subset, we track which sum-ids are used
    
    # Use DFS with bitmask state and memoization on (start_idx, used_sums_mask)
    # But used_sums_mask can be up to 2^m where m can be large
    
    # Alternative: for n=12, just iterate all subsets and check validity
    # 2^12 = 4096 subsets, each check is O(k^2) where k is subset size
    # Total: sum over all subsets of k^2 = O(n^2 * 2^n) = O(144 * 4096) ~ 600K ops
    # That's fast enough
    
    best = 0
    for mask in range(1, 1 << n):
        # Count bits
        k = bin(mask).count('1')
        if k <= best:
            continue
        # Check if this subset is a valid Sidon set
        # Get the elements in the subset
        elems = [i for i in range(n) if mask & (1 << i)]
        # Check all pair sums are unique
        seen_sums = set()
        valid = True
        for idx_a in range(len(elems)):
            i = elems[idx_a]
            for idx_b in range(idx_a, len(elems)):
                j = elems[idx_b]
                s = S[i] + S[j]
                if s in seen_sums:
                    valid = False
                    break
                seen_sums.add(s)
            if not valid:
                break
        if valid:
            best = k
    
    return best


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def _count_duplicate_sums(S):
    """Count the number of duplicate unordered pair sums (including diagonals).
    A 'duplicate' means a sum value that appears more than once among all pairs.
    Higher is better for our purpose (more collisions = smaller max Sidon subset)."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    # Count excess occurrences (sums that appear k times contribute k-1 to duplicates)
    duplicates = sum(v - 1 for v in sum_counts.values() if v > 1)
    return duplicates


def construct():
    start = time.time()
    deadline = start + 87.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        if time.time() > deadline:
            return None
        key = tuple(sorted(c))
        if key in seen:
            return None
        seen.add(key)
        if not _valid(c):
            return None
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(42)

    # Phase 1: Try consecutive sets (known to have small Sidon subsets)
    for lo in range(0, 117):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)

    # Phase 2: Structured sets with many sum collisions
    # Two interleaved APs
    for d in range(1, 8):
        if time.time() > deadline:
            break
        for a in range(0, 100):
            for gap in range(1, 15):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Phase 3: Random sets in small windows
    for trial in range(3000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 90)
        width = rng.randint(12, 25)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Phase 4: Collision-maximization local search
    # Greedily swap elements to maximize the number of duplicate pair sums
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        cur_dup = _count_duplicate_sums(cur)
        
        improved = True
        max_iter = 500
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            # Try all possible single-element swaps
            best_swap = None
            best_swap_dup = cur_dup
            best_swap_val = cur_val
            
            for pos in range(12):
                old_val = cur[pos]
                # Try replacing cur[pos] with a new value
                # Focus on values close to existing elements to create collisions
                candidates = set()
                # Add values near existing elements
                for other in cur:
                    for delta in range(-3, 4):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates.add(nv)
                # Also try some random values
                for _ in range(5):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates.add(nv)
                
                for nv in candidates:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()
                    
                    # Quick check: count duplicates
                    new_dup = _count_duplicate_sums(new_cur)
                    
                    if new_dup > best_swap_dup:
                        best_swap_dup = new_dup
                        best_swap = (pos, nv)
                        # Also compute the actual Sidon size
                        new_val = max_sidon_size(new_cur)
                        best_swap_val = new_val
                        if new_val < cur_val:
                            improved = True
                            break
                    elif new_dup == best_swap_dup:
                        # Tie-breaker: prefer lower Sidon size
                        new_val = max_sidon_size(new_cur)
                        if new_val < best_swap_val:
                            best_swap = (pos, nv)
                            best_swap_val = new_val
                            if new_val < cur_val:
                                improved = True
                                break
                if improved and best_swap_val < cur_val:
                    break
            
            if best_swap is not None:
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                if best_swap_val < cur_val:
                    cur_val = best_swap_val
                    improved = True
                    consider(cur)
        
        # Final consideration
        if cur_val < best_val or (cur_val == best_val and cur_dup > _count_duplicate_sums(best_set)):
            consider(cur)

    # Phase 5: If we have time, try more structured patterns
    # Sets with repeated differences
    if time.time() < deadline - 10:
        for base in range(0, 50):
            if time.time() > deadline - 5:
                break
            for d1 in range(1, 10):
                for d2 in range(1, 10):
                    c = []
                    # 6 elements from AP with step d1, 6 from AP with step d2
                    c1 = [base + i * d1 for i in range(6)]
                    c2 = [base + d2 + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12 and _valid(c):
                        consider(c)

    # Phase 6: Additional collision-maximization search from multiple starting points
    if time.time() < deadline - 15:
        # Start from several different initial sets and do collision-maximization
        start_sets = []
        # Consecutive sets
        for lo in [0, 10, 20, 30, 50, 70]:
            start_sets.append(list(range(lo, lo + 12)))
        # Random sets
        for _ in range(10):
            lo = rng.randint(0, 90)
            width = rng.randint(12, 25)
            hi = min(lo + width, 129)
            if hi - lo >= 12:
                start_sets.append(sorted(rng.sample(range(lo, hi), 12)))
        
        for ss in start_sets:
            if time.time() > deadline - 5:
                break
            if not _valid(ss):
                continue
            cur = list(ss)
            cur_dup = _count_duplicate_sums(cur)
            cur_val = max_sidon_size(cur)
            
            # Quick collision-maximization
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                best_swap = None
                best_swap_dup = cur_dup
                
                for pos in range(12):
                    old_val = cur[pos]
                    candidates = set()
                    for other in cur:
                        for delta in range(-2, 3):
                            nv = other + delta
                            if 0 <= nv <= 128 and nv != old_val:
                                candidates.add(nv)
                    
                    for nv in candidates:
                        if nv in cur:
                            continue
                        new_cur = cur.copy()
                        new_cur[pos] = nv
                        new_cur.sort()
                        new_dup = _count_duplicate_sums(new_cur)
                        if new_dup > best_swap_dup:
                            best_swap_dup = new_dup
                            best_swap = (pos, nv)
                
                if best_swap is None:
                    break
                
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                cur_val = max_sidon_size(cur)
            
            consider(cur)

    # If nothing found, return a default
    if best_set is None:
        best_set = list(range(12))

    return best_set

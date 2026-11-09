"""Two Sidon sets with disjoint difference sets in [1,256].

Strategy:
  1) Fast incremental greedy using bitmask difference sets and sum sets.
  2) Many randomized restarts with different orderings.
  3) A polish pass that tries to insert skipped elements into the best pair.
  4) Local search that swaps elements to free up difference space.
"""

import time
import random

N = 256
MAX_EACH = 64
# Differences range 1..255 -> 255 bits. Use Python int as bitmask.
# Sums range 2..512 -> 511 values; use a set for sums.

def _bitmask_diffs(s):
    """Return bitmask of nonzero differences for sorted list s."""
    mask = 0
    n = len(s)
    for i in range(n):
        for j in range(i):
            d = s[i] - s[j]
            mask |= (1 << d)
    return mask

def _sums_set(s):
    """Return set of pair sums (including doubles) for list s."""
    ss = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            ss.add(s[i] + s[j])
    return ss

def _check_sidon(s):
    """Return True if s is a Sidon set."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _diffs_set(s):
    """Return set of nonzero differences for sorted s."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d

def _valid_pair(A, B):
    """Check both are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs_set(A)
    dB = _diffs_set(B)
    return not (dA & dB)

def _score(A, B):
    return len(A) + len(B)

def _greedy_build_A(pool, target_size, rng):
    """Greedily build a Sidon set A from pool, no diff constraint."""
    result = []
    result_diffs_mask = 0
    result_sums = set()
    pool = list(pool)
    rng.shuffle(pool)
    
    for x in pool:
        if len(result) >= target_size:
            break
        # Check Sidon
        ok = True
        # Check sums: x+a for a in result, and x+x
        for a in result:
            s = a + x
            if s in result_sums:
                ok = False
                break
        if ok:
            if (x + x) in result_sums:
                ok = False
        if ok:
            # Check differences
            for a in result:
                d = abs(x - a)
                if (result_diffs_mask >> d) & 1:
                    ok = False
                    break
            if ok:
                # Add x
                for a in result:
                    result_sums.add(a + x)
                    d = abs(x - a)
                    result_diffs_mask |= (1 << d)
                result_sums.add(x + x)
                result.append(x)
                result.sort()
    
    return result, result_diffs_mask, result_sums

def _greedy_build_B(pool, target_size, A_diffs_mask, rng):
    """Greedily build a Sidon set B from pool, avoiding A's diffs."""
    result = []
    result_diffs_mask = 0
    result_sums = set()
    pool = list(pool)
    rng.shuffle(pool)
    
    for x in pool:
        if len(result) >= target_size:
            break
        # Check Sidon
        ok = True
        for b in result:
            s = b + x
            if s in result_sums:
                ok = False
                break
        if ok:
            if (x + x) in result_sums:
                ok = False
        if ok:
            # Check differences: must not be in A_diffs_mask or B's own diffs
            for b in result:
                d = abs(x - b)
                if ((A_diffs_mask >> d) & 1) or ((result_diffs_mask >> d) & 1):
                    ok = False
                    break
            if ok:
                # Add x
                for b in result:
                    result_sums.add(b + x)
                    d = abs(x - b)
                    result_diffs_mask |= (1 << d)
                result_sums.add(x + x)
                result.append(x)
                result.sort()
    
    return result, result_diffs_mask, result_sums

def _try_insert(A, B, A_diffs_mask, B_diffs_mask, x, side):
    """Try to insert x into side (0=A, 1=B). Returns (success, new_mask, new_sums)."""
    if side == 0:
        if len(A) >= MAX_EACH:
            return False, A_diffs_mask, _sums_set(A)
        other_mask = B_diffs_mask
        self_list = A
        self_mask = A_diffs_mask
    else:
        if len(B) >= MAX_EACH:
            return False, B_diffs_mask, _sums_set(B)
        other_mask = A_diffs_mask
        self_list = B
        self_mask = B_diffs_mask
    
    self_sums = _sums_set(self_list)
    
    # Check Sidon
    ok = True
    for a in self_list:
        s = a + x
        if s in self_sums:
            ok = False
            break
    if ok:
        if (x + x) in self_sums:
            ok = False
    if ok:
        # Check differences
        for a in self_list:
            d = abs(x - a)
            if ((other_mask >> d) & 1) or ((self_mask >> d) & 1):
                ok = False
                break
    if not ok:
        return False, self_mask, self_sums
    
    # Update
    new_mask = self_mask
    new_sums = set(self_sums)
    for a in self_list:
        new_sums.add(a + x)
        d = abs(x - a)
        new_mask |= (1 << d)
    new_sums.add(x + x)
    
    if side == 0:
        return True, new_mask, new_sums
    else:
        return True, new_mask, new_sums

def _polish(A, B, A_diffs_mask, B_diffs_mask, rng, time_limit):
    """Try to insert skipped elements into A or B."""
    A = list(A)
    B = list(B)
    
    # Build set of used elements
    used = set(A) | set(B)
    candidates = [x for x in range(1, N + 1) if x not in used]
    rng.shuffle(candidates)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for x in candidates:
            if time.time() > time_limit:
                break
            # Try adding to A
            ok, new_A_mask, new_A_sums = _try_insert(A, B, A_diffs_mask, B_diffs_mask, x, 0)
            if ok:
                A.append(x)
                A.sort()
                A_diffs_mask = new_A_mask
                improved = True
                candidates.remove(x)
                continue
            # Try adding to B
            ok, new_B_mask, new_B_sums = _try_insert(A, B, A_diffs_mask, B_diffs_mask, x, 1)
            if ok:
                B.append(x)
                B.sort()
                B_diffs_mask = new_B_mask
                improved = True
                candidates.remove(x)
    
    return A, B, A_diffs_mask, B_diffs_mask

def _local_search_swap(A, B, A_diffs_mask, B_diffs_mask, rng, time_limit):
    """Try to improve by swapping elements: remove one from A, add one to B, etc."""
    A = list(A)
    B = list(B)
    
    for _ in range(50):
        if time.time() > time_limit:
            break
        
        # Try removing a random element from A and adding a random candidate to A
        if len(A) > 0:
            idx = rng.randrange(len(A))
            removed = A[idx]
            A_new = A[:idx] + A[idx+1:]
            A_new.sort()
            A_new_mask = _bitmask_diffs(A_new)
            A_new_sums = _sums_set(A_new)
            
            # Try to add some candidate to A_new
            candidates = [x for x in range(1, N + 1) if x not in set(A_new) | set(B)]
            rng.shuffle(candidates)
            for x in candidates[:20]:
                # Check if x can be added to A_new
                ok = True
                for a in A_new:
                    s = a + x
                    if s in A_new_sums:
                        ok = False
                        break
                if ok and (x + x) not in A_new_sums:
                    for a in A_new:
                        d = abs(x - a)
                        if ((B_diffs_mask >> d) & 1) or ((A_new_mask >> d) & 1):
                            ok = False
                            break
                if ok:
                    A_new.append(x)
                    A_new.sort()
                    A_new_mask = _bitmask_diffs(A_new)
                    if len(A_new) > len(A):
                        A = A_new
                        A_diffs_mask = A_new_mask
                    break
    
    return A, B, A_diffs_mask, B_diffs_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s budget (5s margin)
    
    best_A = []
    best_B = []
    best_score = 0
    
    rng = random.Random(42)
    
    # Try various target sizes
    size_pairs = []
    for s in range(15, 45, 3):
        size_pairs.append((s, s))
    for a in range(18, 40, 4):
        for b in range(18, 40, 4):
            size_pairs.append((a, b))
    rng.shuffle(size_pairs)
    
    for A_size, B_size in size_pairs:
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            
            pool = list(range(1, N + 1))
            rng.shuffle(pool)
            
            # Build A
            A, A_mask, A_sums = _greedy_build_A(pool, A_size, rng)
            
            # Build B
            B, B_mask, B_sums = _greedy_build_B(pool, B_size, A_mask, rng)
            
            score = _score(A, B)
            if score > best_score:
                # Verify
                if _valid_pair(A, B):
                    best_A = list(A)
                    best_B = list(B)
                    best_score = score
            
            # Polish
            if score >= best_score - 3 and time.time() < deadline - 3:
                A2, B2, A_mask2, B_mask2 = _polish(A, B, A_mask, B_mask, rng, deadline - 3)
                score2 = _score(A2, B2)
                if score2 > best_score and _valid_pair(A2, B2):
                    best_A = list(A2)
                    best_B = list(B2)
                    best_score = score2
    
    # Final polish on best pair
    if best_score < 2 * MAX_EACH and time.time() < deadline - 2:
        A_mask = _bitmask_diffs(best_A)
        B_mask = _bitmask_diffs(best_B)
        A2, B2, A_mask2, B_mask2 = _polish(best_A, best_B, A_mask, B_mask, rng, deadline - 2)
        score2 = _score(A2, B2)
        if score2 > best_score and _valid_pair(A2, B2):
            best_A = list(A2)
            best_B = list(B2)
            best_score = score2
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final verification
    if not _valid_pair(best_A, best_B):
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96, 192, 255]
        if not _valid_pair(best_A, best_B):
            best_A = [1]
            best_B = [3]
    
    return [best_A, best_B]

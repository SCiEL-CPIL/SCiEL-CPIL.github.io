"""Rewritten program for the suite task; stronger local-search algorithm."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _conflicts(S_set, x):
    """Return set of members c in S such that x + c = c' or c + x = c' or x+x=c' or x+c=x (impossible) with c,c' in S.
    Actually sum-free violation: exists a,b,c in S with a+b=c.
    Adding x to S creates violation if:
      - x + x in S  (a=b=x)
      - x + c in S for some c in S (a=x,b=c or a=c,b=x)
      - c + x in S (same as above)
      - c1 + c2 = x for some c1,c2 in S (then x becomes the 'c')
    So conflicts: 
      if 2x in S -> x conflicts (with itself)
      for c in S: if x+c in S -> x conflicts (and c is involved)
      for c in S: if x in S and ... no.
      Also: if there exist c1,c2 in S with c1+c2 == x, then adding x violates (c1,c2,x).
    """
    conf = set()
    if 2 * x in S_set:
        conf.add(x)  # x with itself
    for c in S_set:
        if x + c in S_set:
            conf.add(x)
            conf.add(c)
    # c1 + c2 = x
    for c1 in S_set:
        for c2 in S_set:
            if c1 + c2 == x:
                conf.add(c1)
                conf.add(c2)
                conf.add(x)
    return conf

def _greedy_large_half(S):
    """Elements > max/2 are sum-free (for positive). But S has negatives.
    Use: sort by value, take elements in top half by value? Not guaranteed sum-free with negatives.
    Safer: use the classic: map to positives via translation? Hard.
    Use greedy: sort by |x| descending, add x if it doesn't conflict.
    """
    S_set = set()
    result = []
    for x in sorted(S, key=lambda v: -abs(v)):
        if _conflicts(S_set, x) == set() or _conflicts(S_set, x) == {x}:
            # only self-conflict means 2x in S_set; if we add x, need 2x not in S_set.
            # _conflicts returns {x} if 2x in S_set -> that's a real conflict (x+x=2x).
            # So we need conflicts to be empty.
            pass
        # Re-evaluate: only add if no conflict at all
        if not _conflicts(S_set, x):
            S_set.add(x)
            result.append(x)
    return result

def _greedy_odd(S):
    return [x for x in S if x % 2 != 0]

def _greedy_large_half_positive(S):
    """For elements, take those > max(S)/2 if all positive; else fallback."""
    mx = max(S)
    mn = min(S)
    # If set is mostly positive, large half works.
    # General: take elements in (max/2, max] — but with negatives this may not be sum-free.
    # Safer: use greedy.
    return _greedy_large_half(S)

def _local_search(S, rng, deadline):
    """Improve a sum-free subset of S via local search."""
    S_set = set(S)
    S_list = list(S)
    all_elements = set(S_list)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    def valid():
        return _is_sum_free(S_set)
    
    # Ensure initial is valid
    if not valid():
        # Fallback: rebuild greedily
        S_set = set(_greedy_large_half(list(S_list)))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
    
    while time.time() < deadline:
        improved = False
        # Try adding each excluded element
        for x in list(excluded):
            if time.time() >= deadline:
                break
            if not _conflicts(S_set, x):
                S_set.add(x)
                excluded.remove(x)
                improved = True
                if len(S_set) > best:
                    best = len(S_set)
                    best_set = set(S_set)
        # Try swaps: add x, remove one conflicting member
        for x in list(excluded):
            if time.time() >= deadline:
                break
            conf = _conflicts(S_set, x)
            if not conf:
                continue
            # Try removing each conflicting member (or subset) to allow x
            for c in list(conf):
                if c == x:
                    continue
                # Try: remove c, add x
                if c in S_set:
                    new_set = (S_set - {c}) | {x}
                    if _is_sum_free(new_set):
                        S_set = new_set
                        excluded.remove(x)
                        improved = True
                        if len(S_set) >= best:
                            if len(S_set) > best:
                                best = len(S_set)
                                best_set = set(S_set)
                        break
        # Try 2-for-1: add two excluded, remove one member
        if len(excluded) >= 2:
            for i in range(len(excluded)):
                if time.time() >= deadline:
                    break
                x1 = excluded[i]
                for j in range(i+1, len(excluded)):
                    x2 = excluded[j]
                    # Try removing each member
                    for c in list(S_set):
                        new_set = (S_set - {c}) | {x1, x2}
                        if _is_sum_free(new_set):
                            S_set = new_set
                            excluded.remove(x1)
                            excluded.remove(x2)
                            improved = True
                            if len(S_set) > best:
                                best = len(S_set)
                                best_set = set(S_set)
                            break
        if not improved:
            # No improvement; try random restarts of local moves
            # Shuffle excluded and retry
            rng.shuffle(excluded)
            # If still stuck, break
            # Try one more pass with different order
            any_move = False
            for x in list(excluded):
                if time.time() >= deadline:
                    break
                if not _conflicts(S_set, x):
                    S_set.add(x)
                    excluded.remove(x)
                    any_move = True
                    if len(S_set) > best:
                        best = len(S_set)
                        best_set = set(S_set)
            if not any_move:
                break
    
    return list(best_set)

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        # Build multiple initial candidates
        cand1 = _greedy_odd(S)
        cand2 = _greedy_large_half(S)
        # Greedy by value descending
        cand3 = []
        S_set3 = set()
        for x in sorted(S, reverse=True):
            if not _conflicts(S_set3, x):
                S_set3.add(x)
                cand3.append(x)
        # Greedy by value ascending
        cand4 = []
        S_set4 = set()
        for x in sorted(S):
            if not _conflicts(S_set4, x):
                S_set4.add(x)
                cand4.append(x)
        
        candidates = [cand1, cand2, cand3, cand4]
        # Pick best initial
        best_init = max(candidates, key=len)
        
        # Local search
        result = _local_search(best_init, rng, deadline)
        
        # Verify
        if not _is_sum_free(result):
            result = best_init
            if not _is_sum_free(result):
                result = cand1
        
        results.append(result)
        
        # Check deadline
        if time.time() > deadline:
            # Fill remaining with greedy
            for size2, seed2 in ((60, 801), (120, 802), (240, 803))[len(results):]:
                S2 = _instance(size2, seed2)
                results.append(_greedy_odd(S2))
            break
    
    return results
# EVOLVE-BLOCK-END

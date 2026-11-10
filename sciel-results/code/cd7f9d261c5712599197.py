"""Maximize sum 1/(512-a) over distinct pairwise-coprime a in 1..511."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

TIME_LIMIT = 100.0
N = 512

def _sieve_primes(limit):
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    return [i for i in range(limit) if sieve[i]]

_PRIMES = _sieve_primes(N)
_P = len(_PRIMES)
_PBIT = [1 << i for i in range(_P)]

def _factor_mask(a):
    """Return bitmask of primes dividing a."""
    m = 0
    x = a
    for i, p in enumerate(_PRIMES):
        if p * p > x:
            break
        if x % p == 0:
            m |= _PBIT[i]
            while x % p == 0:
                x //= p
    if x > 1:
        # x is a prime factor greater than sqrt(original a)
        # find its index in _PRIMES
        for i, p in enumerate(_PRIMES):
            if p == x:
                m |= _PBIT[i]
                break
            if p > x:
                break
    return m

_MASKS = [_factor_mask(a) for a in range(N)]
_SCORES = [1.0 / (N - a) for a in range(N)]  # 1/(512-a)

def _greedy_largest():
    """Greedy: pick largest a first that is coprime to all selected."""
    mask = 0
    sel = []
    for a in range(N - 1, 0, -1):
        fm = _MASKS[a]
        if (mask & fm) == 0:
            mask |= fm
            sel.append(a)
    return sel, mask

def _try_swap(sel, mask):
    """Try to improve by swapping out a selected element for a better one, or 2-for-1."""
    selset = set(sel)
    improved = True
    while improved:
        improved = False
        # 1-for-1 swaps
        for idx, a in enumerate(sel):
            fm_a = _MASKS[a]
            base_mask = mask & ~fm_a
            best_b = None
            best_score = _SCORES[a]
            for b in range(1, N):
                if b in selset:
                    continue
                fm_b = _MASKS[b]
                if (base_mask & fm_b) != 0:
                    continue
                if _SCORES[b] > best_score:
                    best_score = _SCORES[b]
                    best_b = b
            if best_b is not None:
                sel[idx] = best_b
                selset.discard(a)
                selset.add(best_b)
                mask = base_mask | _MASKS[best_b]
                improved = True
                break
        if improved:
            continue
        # 2-for-1: remove two elements, add one better
        n = len(sel)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = sel[i], sel[j]
                fm1, fm2 = _MASKS[a1], _MASKS[a2]
                base_mask = mask & ~(fm1 | fm2)
                removed_score = _SCORES[a1] + _SCORES[a2]
                best_b = None
                best_b_score = removed_score
                for b in range(1, N):
                    if b in selset:
                        continue
                    fm_b = _MASKS[b]
                    if (base_mask & fm_b) != 0:
                        continue
                    if _SCORES[b] > best_b_score:
                        best_b_score = _SCORES[b]
                        best_b = b
                if best_b is not None:
                    new_sel = sel[:i] + sel[i+1:j] + sel[j+1:]
                    new_sel.append(best_b)
                    new_selset = set(new_sel)
                    new_mask = base_mask | _MASKS[best_b]
                    sel = new_sel
                    selset = new_selset
                    mask = new_mask
                    improved = True
                    break
            if improved:
                break
    return sel, mask

def _try_double_swap(sel, mask):
    """Try to improve by removing two elements and adding one or two better ones."""
    selset = set(sel)
    improved = True
    while improved:
        improved = False
        n = len(sel)
        for i in range(n):
            for j in range(i+1, n):
                a1, a2 = sel[i], sel[j]
                fm1, fm2 = _MASKS[a1], _MASKS[a2]
                base_mask = mask & ~(fm1 | fm2)
                removed_score = _SCORES[a1] + _SCORES[a2]
                
                # Try adding one element that replaces both
                best_single = None
                best_single_score = removed_score
                for b in range(1, N):
                    if b in selset:
                        continue
                    fm_b = _MASKS[b]
                    if (base_mask & fm_b) != 0:
                        continue
                    if _SCORES[b] > best_single_score:
                        best_single_score = _SCORES[b]
                        best_single = b
                
                if best_single is not None:
                    new_sel = sel[:i] + sel[i+1:j] + sel[j+1:]
                    new_selset = set(new_sel)
                    new_mask = base_mask | _MASKS[best_single]
                    new_score = sum(_SCORES[x] for x in new_sel) + _SCORES[best_single]
                    if new_score > sum(_SCORES[x] for x in sel) + 1e-15:
                        sel = new_sel
                        selset = new_selset
                        mask = new_mask
                        improved = True
                        break
                
                # Try adding two elements that replace both
                best_pair = None
                best_pair_score = removed_score
                for b1 in range(1, N):
                    if b1 in selset:
                        continue
                    fm_b1 = _MASKS[b1]
                    if (base_mask & fm_b1) != 0:
                        continue
                    for b2 in range(b1+1, N):
                        if b2 in selset:
                            continue
                        fm_b2 = _MASKS[b2]
                        if (base_mask & fm_b2) != 0:
                            continue
                        if (fm_b1 & fm_b2) != 0:
                            continue
                        pair_score = _SCORES[b1] + _SCORES[b2]
                        if pair_score > best_pair_score:
                            best_pair_score = pair_score
                            best_pair = (b1, b2)
                
                if best_pair is not None:
                    b1, b2 = best_pair
                    new_sel = sel[:i] + sel[i+1:j] + sel[j+1:]
                    new_sel.append(b1)
                    new_sel.append(b2)
                    new_selset = set(new_sel)
                    new_mask = base_mask | _MASKS[b1] | _MASKS[b2]
                    new_score = sum(_SCORES[x] for x in new_sel)
                    if new_score > sum(_SCORES[x] for x in sel) + 1e-15:
                        sel = new_sel
                        selset = new_selset
                        mask = new_mask
                        improved = True
                        break
            if improved:
                break
    return sel, mask

def _try_add(sel, mask):
    """Try to add more elements that are coprime to all selected."""
    selset = set(sel)
    improved = True
    while improved:
        improved = False
        for b in range(N - 1, 0, -1):
            if b in selset:
                continue
            fm_b = _MASKS[b]
            if (mask & fm_b) == 0:
                sel.append(b)
                selset.add(b)
                mask |= fm_b
                improved = True
    return sel, mask

def _exact_score(sel):
    total = Fraction(0, 1)
    for a in sel:
        total += Fraction(1, N - a)
    return float(total)

def construct():
    t0 = time.time()
    
    # Start with greedy from largest
    sel, mask = _greedy_largest()
    sel, mask = _try_add(sel, mask)
    sel, mask = _try_swap(sel, mask)
    sel, mask = _try_double_swap(sel, mask)
    
    best_sel = sel[:]
    best_score = _exact_score(best_sel)
    
    # Local search: random restarts with perturbation
    import random
    rng = random.Random(42)
    
    while time.time() - t0 < TIME_LIMIT - 2.0:
        # Perturb: remove 1-4 random elements
        cur_sel = best_sel[:]
        cur_mask = 0
        for a in cur_sel:
            cur_mask |= _MASKS[a]
        
        k = 1 + rng.randint(0, 3)
        for _ in range(k):
            if len(cur_sel) <= 1:
                break
            idx = rng.randrange(len(cur_sel))
            a = cur_sel.pop(idx)
            cur_mask &= ~_MASKS[a]
        
        # Re-greedy: add back largest possible
        cur_sel, cur_mask = _try_add(cur_sel, cur_mask)
        cur_sel, cur_mask = _try_swap(cur_sel, cur_mask)
        cur_sel, cur_mask = _try_double_swap(cur_sel, cur_mask)
        
        cur_score = _exact_score(cur_sel)
        if cur_score > best_score + 1e-15:
            best_sel = cur_sel
            best_score = cur_score
    
    return best_sel
# EVOLVE-BLOCK-END

"""Rewritten program: greedy + hill-climb with bitmask coprimality for pairwise-coprime selection."""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 145.0
_START = None

def _elapsed():
    return time.time() - _START if _START is not None else 0.0

def _deadline_guard(limit=_DEADLINE, margin=2.0):
    return _elapsed() > (limit - margin)

def _sieve_primes(n):
    """Return list of primes < n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n - 1)) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = 0
    return [i for i in range(2, n) if sieve[i]]

def _build_prime_masks(primes):
    """For each number 1..511, compute bitmask of its prime factors."""
    masks = [0] * 512
    for a in range(2, 512):
        x = a
        mask = 0
        for idx, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << idx)
                while x % p == 0:
                    x //= p
        if x > 1:
            for idx, p in enumerate(primes):
                if p == x:
                    mask |= (1 << idx)
                    break
        masks[a] = mask
    return masks

def _score(chosen):
    return sum(1.0 / (512.0 - a) for a in chosen)

def _greedy_select(masks, order):
    """Greedy select in given order using bitmask coprimality."""
    chosen = []
    chosen_mask = 0
    for a in order:
        if masks[a] & chosen_mask:
            continue
        chosen.append(a)
        chosen_mask |= masks[a]
    return chosen

def _hill_climb(chosen, masks, weights):
    """Optimized Hill-climbing: 1-add, 1-swap, and heuristic 2-swap (remove 1, add 2)."""
    chosen_set = set(chosen)
    cur_score = sum(weights[a] for a in chosen)
    
    improved = True
    while improved:
        if _deadline_guard():
            break
        improved = False
        best_gain = 0.0
        best_action = None
        
        # 1. Try 1-Add
        chosen_mask = 0
        for a in chosen:
            chosen_mask |= masks[a]
            
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if masks[a] & chosen_mask:
                continue
            gain = weights[a]
            if gain > best_gain:
                best_gain = gain
                best_action = ('add', a)
        
        # 2. Try 1-Swap (remove r, add a)
        for r in chosen:
            if _deadline_guard():
                break
            mask_others = 0
            for c in chosen:
                if c != r:
                    mask_others |= masks[c]
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & mask_others:
                    continue
                gain = weights[a] - weights[r]
                if gain > best_gain:
                    best_gain = gain
                    best_action = ('swap', r, a)
                    
        # 3. Try 2-Swap (remove r, add a1, a2) - Heuristic
        for r in chosen:
            if _deadline_guard():
                break
            mask_others = 0
            for c in chosen:
                if c != r:
                    mask_others |= masks[c]
            
            candidates = []
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & mask_others:
                    continue
                candidates.append(a)
                if len(candidates) > 15:
                    break
            
            if len(candidates) >= 2:
                for i in range(len(candidates)):
                    a1 = candidates[i]
                    m1 = masks[a1]
                    for j in range(i + 1, len(candidates)):
                        a2 = candidates[j]
                        if m1 & masks[a2]:
                            continue
                        gain = weights[a1] + weights[a2] - weights[r]
                        if gain > best_gain:
                            best_gain = gain
                            best_action = ('swap2', r, a1, a2)
        
        if best_action is not None and best_gain > 1e-15:
            if best_action[0] == 'add':
                a = best_action[1]
                chosen.append(a)
                chosen_set.add(a)
                cur_score += weights[a]
            elif best_action[0] == 'swap':
                r, a = best_action[1], best_action[2]
                chosen.remove(r)
                chosen_set.remove(r)
                chosen.append(a)
                chosen_set.add(a)
                cur_score += weights[a] - weights[r]
            elif best_action[0] == 'swap2':
                r, a1, a2 = best_action[1], best_action[2], best_action[3]
                chosen.remove(r)
                chosen_set.remove(r)
                chosen.append(a1)
                chosen.append(a2)
                chosen_set.add(a1)
                chosen_set.add(a2)
                cur_score += weights[a1] + weights[a2] - weights[r]
            improved = True
            
    return chosen, cur_score

def construct():
    global _START
    _START = time.time()
    
    N = 512
    primes = _sieve_primes(N)
    masks = _build_prime_masks(primes)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    best_chosen = []
    best_score = 0.0
    rng = random.Random(42)
    
    # Multiple greedy initializations with different orderings
    n_restarts = 200
    for restart in range(n_restarts):
        if _deadline_guard():
            break
        
        if restart == 0:
            order = list(range(511, 0, -1))
        elif restart == 1:
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        elif restart == 2:
            order = sorted(range(511, 0, -1), key=lambda a: (0 if masks[a].bit_count() == 1 else 1, -a))
        elif restart % 2 == 0:
            order = list(range(511, 0, -1))
            rng.shuffle(order)
        else:
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(5, 150)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        
        # Hill-climb to local optimum
        chosen, score = _hill_climb(chosen, masks, weights)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
    
    # Verify
    assert len(set(best_chosen)) == len(best_chosen)
    assert all(1 <= x <= 511 for x in best_chosen)
    for i in range(len(best_chosen)):
        for j in range(i + 1, len(best_chosen)):
            assert math.gcd(best_chosen[i], best_chosen[j]) == 1
    
    return best_chosen
# EVOLVE-BLOCK-END

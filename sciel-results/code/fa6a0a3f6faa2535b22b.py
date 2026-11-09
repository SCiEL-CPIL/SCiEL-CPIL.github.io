"""Combined: bitmask prime-factor coprimality + multi-restart greedy + hill-climb with 2/3-swap + final polish."""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 145.0
_START = None

def _elapsed():
    return time.time() - _START if _START is not None else 0.0

def _deadline_guard(margin=3.0):
    return _elapsed() > (_DEADLINE - margin)

def _sieve_primes(n):
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
    chosen = []
    chosen_mask = 0
    for a in order:
        if masks[a] & chosen_mask:
            continue
        chosen.append(a)
        chosen_mask |= masks[a]
    return chosen

def _is_compatible(a, chosen_mask, masks):
    return (masks[a] & chosen_mask) == 0

def _hill_climb(chosen, masks, max_iters=200):
    chosen_set = set(chosen)
    cur_score = _score(chosen)
    chosen_mask = 0
    for a in chosen:
        chosen_mask |= masks[a]
    
    iters = 0
    improved = True
    while improved and iters < max_iters:
        iters += 1
        if _deadline_guard():
            break
        improved = False
        best_gain = 0.0
        best_action = None
        
        # Try adding each candidate not in chosen
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if masks[a] & chosen_mask:
                continue
            gain = 1.0 / (512.0 - a)
            if gain > best_gain:
                best_gain = gain
                best_action = ('add', a, None)
        
        # Try 2-swap: remove one chosen element, add one new element
        for rem in chosen:
            if _deadline_guard():
                break
            rem_chosen_mask = 0
            for a in chosen:
                if a != rem:
                    rem_chosen_mask |= masks[a]
            
            rem_loss = 1.0 / (512.0 - rem)
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & rem_chosen_mask:
                    continue
                gain = 1.0 / (512.0 - a) - rem_loss
                if gain > best_gain:
                    best_gain = gain
                    best_action = ('swap', a, rem)
        
        # Try 3-swap: remove two chosen, add one new (net gain if new is much better)
        n_chosen = len(chosen)
        if n_chosen >= 2:
            for i in range(n_chosen):
                for j in range(i + 1, n_chosen):
                    if _deadline_guard():
                        break
                    rem1, rem2 = chosen[i], chosen[j]
                    rem_chosen_mask = 0
                    for a in chosen:
                        if a != rem1 and a != rem2:
                            rem_chosen_mask |= masks[a]
                    
                    rem_loss = 1.0 / (512.0 - rem1) + 1.0 / (512.0 - rem2)
                    
                    for a in range(511, 0, -1):
                        if a in chosen_set:
                            continue
                        if masks[a] & rem_chosen_mask:
                            continue
                        gain = 1.0 / (512.0 - a) - rem_loss
                        if gain > best_gain:
                            best_gain = gain
                            best_action = ('swap2', a, rem1, rem2)
        
        if best_action is not None and best_gain > 1e-15:
            if best_action[0] == 'add':
                a = best_action[1]
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= masks[a]
                cur_score += best_gain
            elif best_action[0] == 'swap':
                a, rem = best_action[1], best_action[2]
                chosen.remove(rem)
                chosen_set.discard(rem)
                chosen_mask &= ~masks[rem]
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= masks[a]
                cur_score += best_gain
            elif best_action[0] == 'swap2':
                a, rem1, rem2 = best_action[1], best_action[2], best_action[3]
                chosen.remove(rem1)
                chosen.remove(rem2)
                chosen_set.discard(rem1)
                chosen_set.discard(rem2)
                chosen_mask &= ~masks[rem1]
                chosen_mask &= ~masks[rem2]
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= masks[a]
                cur_score += best_gain
            improved = True
    
    return chosen, cur_score

def _final_polish(chosen, masks):
    """Systematic pairwise-exchange polish: try all (remove, add) pairs."""
    chosen_set = set(chosen)
    cur_score = _score(chosen)
    chosen_mask = 0
    for a in chosen:
        chosen_mask |= masks[a]
    
    improved = True
    while improved and not _deadline_guard(5.0):
        improved = False
        best_gain = 0.0
        best_rem = None
        best_add = None
        
        for rem in chosen:
            rem_chosen_mask = 0
            for a in chosen:
                if a != rem:
                    rem_chosen_mask |= masks[a]
            rem_loss = 1.0 / (512.0 - rem)
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & rem_chosen_mask:
                    continue
                gain = 1.0 / (512.0 - a) - rem_loss
                if gain > best_gain:
                    best_gain = gain
                    best_rem = rem
                    best_add = a
        
        if best_gain > 1e-12 and best_rem is not None:
            chosen.remove(best_rem)
            chosen_set.discard(best_rem)
            chosen_mask &= ~masks[best_rem]
            chosen.append(best_add)
            chosen_set.add(best_add)
            chosen_mask |= masks[best_add]
            cur_score += best_gain
            improved = True
    
    return chosen, cur_score

def construct():
    global _START
    _START = time.time()
    
    N = 512
    primes = _sieve_primes(N)
    masks = _build_prime_masks(primes)
    
    best_chosen = []
    best_score = 0.0
    rng = random.Random(12345)
    
    # Phase 1: Multiple greedy initializations with hill-climbing
    n_restarts = 50
    for restart in range(n_restarts):
        if _deadline_guard(10.0):
            break
        
        if restart == 0:
            order = list(range(511, 0, -1))
        elif restart == 1:
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        elif restart == 2:
            order = sorted(range(511, 0, -1), key=lambda a: (-a, masks[a].bit_count()))
        elif restart == 3:
            # Prefer numbers with exactly one prime factor (primes and prime powers)
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        else:
            # Perturbed descending order
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(10, 200)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        chosen, score = _hill_climb(chosen, masks)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
    
    # Phase 2: Final polish on best solution
    best_chosen, best_score = _final_polish(best_chosen, masks)
    
    # Phase 3: If time remains, try more random restarts with the best as seed
    if not _deadline_guard(15.0):
        for extra_restart in range(30):
            if _deadline_guard(5.0):
                break
            
            # Start from best, remove 1-3 random elements, re-greedy fill
            seed = best_chosen[:]
            n_remove = rng.randint(1, 3)
            for _ in range(n_remove):
                if seed:
                    idx = rng.randint(0, len(seed) - 1)
                    seed.pop(idx)
            
            # Greedy fill from seed
            seed_set = set(seed)
            seed_mask = 0
            for a in seed:
                seed_mask |= masks[a]
            
            for a in range(511, 0, -1):
                if a in seed_set:
                    continue
                if masks[a] & seed_mask:
                    continue
                seed.append(a)
                seed_set.add(a)
                seed_mask |= masks[a]
            
            seed, score = _hill_climb(seed, masks)
            
            if score > best_score:
                best_score = score
                best_chosen = seed[:]
        
        # Final polish again
        best_chosen, best_score = _final_polish(best_chosen, masks)
    
    # Verify
    assert len(set(best_chosen)) == len(best_chosen)
    assert all(1 <= x <= 511 for x in best_chosen)
    for i in range(len(best_chosen)):
        for j in range(i + 1, len(best_chosen)):
            assert math.gcd(best_chosen[i], best_chosen[j]) == 1
    
    return best_chosen
# EVOLVE-BLOCK-END

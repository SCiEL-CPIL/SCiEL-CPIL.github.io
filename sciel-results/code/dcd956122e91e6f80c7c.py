"""Football pool / ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130
    
    total = q ** n
    rng = random.Random(42)
    
    # Precompute neighbors as list of arrays
    neighbors = []
    for w in range(total):
        digits = []
        tmp = w
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1]
        nb = [w]
        for pos in range(n):
            orig = digits[pos]
            for v in range(q):
                if v == orig:
                    continue
                nd = digits[:]
                nd[pos] = v
                idx = 0
                for d in nd:
                    idx = idx * q + d
                nb.append(idx)
        neighbors.append(nb)
    
    def idx_to_digits(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        return digits[::-1]
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx
    
    # Initial: tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            full = h + list(tail)
            initial.append(digits_to_idx(full))
    
    best_chosen = list(initial)
    best_size = len(initial)
    
    def greedy_cover(start_set, deadline, rng, tie_mode=0):
        """Greedy set cover with incremental gain tracking."""
        covered = np.zeros(total, dtype=np.int32)
        chosen = list(start_set)
        chosen_set = set(chosen)
        for c in chosen:
            for w in neighbors[c]:
                covered[w] += 1
        
        # gain[c] = number of uncovered words that c covers
        gain = np.zeros(total, dtype=np.int32)
        for c in range(total):
            for w in neighbors[c]:
                if covered[w] == 0:
                    gain[c] += 1
        for c in chosen_set:
            gain[c] = -1
        
        uncovered_count = total - int(np.count_nonzero(covered))
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            if tie_mode == 0:
                best = int(np.argmax(gain))
            elif tie_mode == 1:
                candidates = np.where(gain == max_gain)[0]
                best = int(rng.choice(candidates))
            else:
                candidates = np.where(gain == max_gain)[0]
                best = int(candidates[0])
            chosen.append(best)
            chosen_set.add(best)
            # Update coverage and gains
            for w in neighbors[best]:
                if covered[w] == 0:
                    uncovered_count -= 1
                    # This word was uncovered, so all its neighbors lose gain
                    for c in neighbors[w]:
                        if c not in chosen_set:
                            gain[c] -= 1
                covered[w] += 1
            gain[best] = -1
        return chosen
    
    def try_remove_redundant(chosen, deadline):
        covered = np.zeros(total, dtype=np.int32)
        for c in chosen:
            for w in neighbors[c]:
                covered[w] += 1
        new_chosen = list(chosen)
        for c in chosen[:]:
            if time.time() > deadline:
                break
            redundant = True
            for w in neighbors[c]:
                if covered[w] <= 1:
                    redundant = False
                    break
            if redundant:
                for w in neighbors[c]:
                    covered[w] -= 1
                new_chosen.remove(c)
        return new_chosen
    
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        if attempt <= 2:
            start_set = initial
            tie_mode = 0
        elif attempt <= 4:
            start_set = initial
            tie_mode = attempt % 3
        else:
            frac = 0.4 + 0.4 * rng.random()
            k = max(1, int(len(initial) * frac))
            start_set = rng.sample(initial, k)
            tie_mode = attempt % 3
        
        chosen = greedy_cover(start_set, deadline, rng, tie_mode)
        
        if time.time() < deadline - 3:
            chosen = try_remove_redundant(chosen, deadline)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = chosen
            if time.time() < deadline - 3:
                best_chosen = try_remove_redundant(best_chosen, deadline)
                best_size = len(best_chosen)
        
        if best_size <= 403:
            break
    
    result = [idx_to_digits(idx) for idx in best_chosen]
    return result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 135
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_t = len(all_triples)  # 560
    
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute completion sets as bitmasks
    # For each triple idx, list of bitmasks (each bitmask represents the 7 other triples)
    completion_masks = [[] for _ in range(num_t)]
    
    for idx, (a, b, c) in enumerate(all_triples):
        for a_prime in range(n):
            if a_prime in (a, b, c):
                continue
            for b_prime in range(n):
                if b_prime in (a, b, c, a_prime):
                    continue
                for c_prime in range(n):
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    others = [
                        (a_prime, b, c),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a, b, c_prime),
                        (a_prime, b, c_prime),
                    ]
                    mask = 0
                    valid = True
                    for o in others:
                        if o in triple_idx:
                            mask |= (1 << triple_idx[o])
                        else:
                            valid = False
                            break
                    if valid:
                        completion_masks[idx].append(mask)
    
    def can_add_bitmask(idx, present_mask):
        for mask in completion_masks[idx]:
            if (present_mask & mask) == mask:
                return False
        return True
    
    def greedy(seed_val):
        random.seed(seed_val)
        order = list(range(num_t))
        random.shuffle(order)
        present_mask = 0
        count = 0
        for idx in order:
            if can_add_bitmask(idx, present_mask):
                present_mask |= (1 << idx)
                count += 1
        return present_mask, count
    
    def local_search_bitmask(present_mask, max_rounds=50):
        present_set = set()
        for i in range(num_t):
            if (present_mask >> i) & 1:
                present_set.add(i)
        excluded = set(range(num_t)) - present_set
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for idx in list(excluded):
                if can_add_bitmask(idx, present_mask):
                    present_mask |= (1 << idx)
                    present_set.add(idx)
                    excluded.discard(idx)
                    improved = True
                    break
            if improved:
                continue
            for idx_rem in list(present_set):
                for idx_add in list(excluded):
                    new_mask = present_mask & ~(1 << idx_rem)
                    if can_add_bitmask(idx_add, new_mask):
                        present_mask = new_mask | (1 << idx_add)
                        present_set.discard(idx_rem)
                        present_set.add(idx_add)
                        excluded.discard(idx_add)
                        excluded.add(idx_rem)
                        improved = True
                        break
                if improved:
                    break
        return present_mask
    
    best_mask = 0
    best_size = 0
    
    start = time.time()
    seed_counter = 0
    while time.time() < deadline:
        if time.time() - start > 130:
            break
        mask, cnt = greedy(seed_counter)
        if cnt > best_size:
            best_size = cnt
            best_mask = mask
        ls_start = time.time()
        mask2 = local_search_bitmask(mask, max_rounds=30)
        cnt2 = bin(mask2).count('1')
        if cnt2 > best_size:
            best_size = cnt2
            best_mask = mask2
        seed_counter += 1
        if time.time() - ls_start > 120:
            break
    
    result = []
    for i in range(num_t):
        if (best_mask >> i) & 1:
            result.append(list(all_triples[i]))
    return result
# EVOLVE-BLOCK-END

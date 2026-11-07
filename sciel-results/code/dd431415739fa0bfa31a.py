"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

def construct():
    n_elements = 12
    universe_size = 1 << n_elements  # 4096
    family_size = 40

    all_masks = list(range(universe_size))

    def count_comparable(family):
        count = 0
        n = len(family)
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    # Greedy construction: start with the empty set and build up by adding the
    # mask that maximizes the number of new comparable pairs it forms with the
    # existing family.
    random.seed(42)
    
    # Start with a random small seed family to break symmetry, then grow greedily
    current_family = [0]  # Start with empty set
    
    # Precompute subset/superset relationships for efficiency
    # For each mask, we could check against all others, but let's use a smarter approach
    # We'll maintain the family and for each candidate, compute how many existing members
    # it is comparable to.

    def comparable_count_with_family(mask, family):
        count = 0
        for existing in family:
            if mask == existing:
                continue
            if (mask & existing) == mask or (mask & existing) == existing:
                count += 1
        return count

    # Greedily add the best mask
    for _ in range(family_size - 1):
        best_mask = -1
        best_gain = -1
        # Try all masks not already in family
        for mask in all_masks:
            if mask in current_family:
                continue
            gain = comparable_count_with_family(mask, current_family)
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask == -1:
            # Should not happen since universe is much larger
            break
        current_family.append(best_mask)

    best_family = current_family.copy()
    best_score = count_comparable(best_family)

    # Now do local search / simulated annealing to improve
    start_time = time.time()
    timeout = 55
    temperature = 10.0
    cooling = 0.9995

    while time.time() - start_time < timeout:
        idx = random.randint(0, family_size - 1)
        old_set = best_family[idx]

        if random.random() < 0.8:
            bit_to_flip = random.randint(0, n_elements - 1)
            new_set = old_set ^ (1 << bit_to_flip)
        else:
            new_set = random.choice(all_masks)

        if new_set == old_set:
            continue

        is_dup = False
        for k in range(family_size):
            if k != idx and best_family[k] == new_set:
                is_dup = True
                break

        if is_dup:
            continue

        candidate_family = best_family.copy()
        candidate_family[idx] = new_set
        candidate_score = count_comparable(candidate_family)

        delta = candidate_score - best_score

        if delta > 0 or random.random() < math.exp(delta / temperature):
            best_family = candidate_family
            best_score = candidate_score

        temperature *= cooling
        if temperature < 0.05:
            temperature = 0.05

    result = []
    for mask in best_family:
        subset = []
        for i in range(n_elements):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)

    return result
# EVOLVE-BLOCK-END

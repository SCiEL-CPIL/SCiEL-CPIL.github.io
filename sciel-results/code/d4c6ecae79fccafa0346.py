import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    # Each config is defined by 3 disjoint pairs, yielding 8 transversal triples
    triple_to_kconfigs = {t: [] for t in all_triples}
    kconfigs = []
    
    def partitions_into_pairs(points):
        a = points[0]
        for i in range(1, 6):
            b = points[i]
            remaining = [p for j, p in enumerate(points) if j not in (0, i)]
            c, d, e, f = remaining
            yield (a, b), (c, d), (e, f)
            yield (a, b), (c, e), (d, f)
            yield (a, b), (c, f), (d, e)
    
    for six_points in combinations(range(n), 6):
        for pairs in partitions_into_pairs(six_points):
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            transversals_canon = set()
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        transversals_canon.add(tuple(sorted((x, y, z))))
            kconfigs.append(transversals_canon)
            for t in transversals_canon:
                triple_to_kconfigs[t].append(len(kconfigs) - 1)
    
    num_kconfigs = len(kconfigs)
    
    def greedy_solve(order):
        present = set()
        kconfig_counts = [0] * num_kconfigs
        for t in order:
            can_add = True
            for kidx in triple_to_kconfigs[t]:
                if kconfig_counts[kidx] == 7:
                    can_add = False
                    break
            if not can_add:
                continue
            present.add(t)
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] += 1
        return present, kconfig_counts
        
    def local_improve(present, kconfig_counts):
        # Try to add any missing triple
        improved = True
        while improved:
            improved = False
            for t in all_triples:
                if t in present:
                    continue
                can_add = True
                for kidx in triple_to_kconfigs[t]:
                    if kconfig_counts[kidx] == 7:
                        can_add = False
                        break
                if can_add:
                    present.add(t)
                    for kidx in triple_to_kconfigs[t]:
                        kconfig_counts[kidx] += 1
                    improved = True
        return present, kconfig_counts

    # Initial best solution
    best_count = 0
    best_triples = None
    
    # Warm start with greedy
    random.seed(42)
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    present, kconfig_counts = greedy_solve(shuffled)
    present, kconfig_counts = local_improve(present, kconfig_counts)
    best_count = len(present)
    best_triples = [list(t) for t in present]
    
    # ILS Search
    start_time = time.time()
    time_limit = 110.0
    
    current_present = set(present)
    current_kconfig_counts = kconfig_counts[:]
    
    while time.time() - start_time < time_limit:
        # Perturb: remove k random triples
        k = random.randint(2, 5)
        to_remove = random.sample(list(current_present), min(k, len(current_present)))
        
        # Create a copy to perturb
        perturbed_present = set(current_present)
        perturbed_kconfig_counts = current_kconfig_counts[:]
        
        for t in to_remove:
            perturbed_present.remove(t)
            for kidx in triple_to_kconfigs[t]:
                perturbed_kconfig_counts[kidx] -= 1
        
        # Recover: local improve
        perturbed_present, perturbed_kconfig_counts = local_improve(perturbed_present, perturbed_kconfig_counts)
        
        # Check if better
        if len(perturbed_present) > best_count:
            best_count = len(perturbed_present)
            best_triples = [list(t) for t in perturbed_present]
            
        # Update current state to the perturbed one (accepting worse solutions is standard in ILS)
        current_present = perturbed_present
        current_kconfig_counts = perturbed_kconfig_counts
        
    return best_triples

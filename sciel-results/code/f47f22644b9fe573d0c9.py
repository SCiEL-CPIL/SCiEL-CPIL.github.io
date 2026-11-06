import random
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
        return present
    
    def local_improve(present):
        # Try to add any missing triple
        improved = True
        while improved:
            improved = False
            kconfig_counts = [0] * num_kconfigs
            for t in present:
                for kidx in triple_to_kconfigs[t]:
                    kconfig_counts[kidx] += 1
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
        return present
    
    best_count = 0
    best_triples = None
    
    for trial in range(30):
        random.seed(trial)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        present = greedy_solve(shuffled)
        present = local_improve(present)
        
        if len(present) > best_count:
            best_count = len(present)
            best_triples = [list(t) for t in present]
    
    return best_triples

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    all_elements = list(range(1, N + 1))
    
    # Pre-compute LCM table for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, N + 1))
    
    # Build conflict map: for each element v, the set of pairs (u, w) such that {v, u, w} is bad
    conflict_map = defaultdict(set)
    degree = defaultdict(int)
    for a, b, c in bad_triples:
        conflict_map[a].add((b, c))
        conflict_map[b].add((a, c))
        conflict_map[c].add((a, b))
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Helper: check if adding element v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                return False
        return True
    
    # Strategy: Greedy Independent Set on Conflict Hypergraph with multiple orderings
    # Initial greedy: sort by degree ascending (least constrained first)
    sorted_elements = sorted(all_elements, key=lambda x: degree[x])
    
    selected_set = set()
    for v in sorted_elements:
        if time.time() > deadline:
            break
        if can_add(v, selected_set):
            selected_set.add(v)
            
    best_set = set(selected_set)
    best_size = len(best_set)
    
    # Multiple random restarts with diverse ordering strategies
    random.seed(42)
    
    for restart in range(150):
        if time.time() > deadline:
            break
        
        # Cycle through different ordering strategies
        strategy = restart % 5
        if strategy == 0:
            # Degree ascending with noise
            perturbed_elements = sorted(all_elements, key=lambda x: degree[x] + random.random())
        elif strategy == 1:
            # Degree descending with noise
            perturbed_elements = sorted(all_elements, key=lambda x: -degree[x] + random.random())
        elif strategy == 2:
            # Pure random
            perturbed_elements = all_elements[:]
            random.shuffle(perturbed_elements)
        elif strategy == 3:
            # Bucket sort by degree with random within buckets
            buckets = defaultdict(list)
            for x in all_elements:
                buckets[degree[x]].append(x)
            perturbed_elements = []
            for d in sorted(buckets.keys()):
                bucket = buckets[d]
                random.shuffle(bucket)
                perturbed_elements.extend(bucket)
        else:
            # Weighted random based on inverse degree
            weights = [1.0 / (degree[x] + 1) for x in all_elements]
            perturbed_elements = []
            remaining = all_elements[:]
            remaining_weights = weights[:]
            while remaining:
                total_w = sum(remaining_weights)
                r = random.random() * total_w
                cum = 0
                for i, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        perturbed_elements.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break
        
        current_set = set()
        for v in perturbed_elements:
            if can_add(v, current_set):
                current_set.add(v)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            
    # Local search: 1-for-2 swaps (remove one, add two)
    if time.time() < deadline - 3.0:
        current_best = set(best_set)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            for v in list(current_best):
                if time.time() > deadline - 1.0:
                    break
                test_set = current_best - {v}
                candidates = []
                for w in all_elements:
                    if w not in test_set and can_add(w, test_set):
                        candidates.append(w)
                        
                if len(candidates) >= 2:
                    # Try pairs in order of lowest degree first
                    candidates.sort(key=lambda x: degree[x])
                    found_pair = False
                    for i in range(min(len(candidates), 20)):  # Limit to top 20
                        for j in range(i + 1, min(len(candidates), 20)):
                            w1, w2 = candidates[i], candidates[j]
                            if can_add(w1, test_set) and can_add(w2, test_set | {w1}):
                                new_set = test_set | {w1, w2}
                                if len(new_set) > len(current_best):
                                    current_best = new_set
                                    improved = True
                                    found_pair = True
                                    break
                        if found_pair:
                            break
                                    
        if len(current_best) > len(best_set):
            best_set = current_best
            best_size = len(best_set)

    # Ensure non-empty
    if not best_set:
        best_set = {1}
        
    return list(best_set)

# EVOLVE-BLOCK-END

import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations and the conflict graph
    # A K(2,2,2) is defined by three disjoint pairs
    # For each set of 6 points, generate all 15 perfect matchings
    conflict = [set() for _ in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        # Generate all perfect matchings of 6 points
        a, b, c, d, e, f = six_points
        matchings = [
            [(a,b),(c,d),(e,f)],
            [(a,b),(c,e),(d,f)],
            [(a,b),(c,f),(d,e)],
            [(a,c),(b,d),(e,f)],
            [(a,c),(b,e),(d,f)],
            [(a,c),(b,f),(d,e)],
            [(a,d),(b,c),(e,f)],
            [(a,d),(b,e),(c,f)],
            [(a,d),(b,f),(c,e)],
            [(a,e),(b,c),(d,f)],
            [(a,e),(b,d),(c,f)],
            [(a,e),(b,f),(c,d)],
            [(a,f),(b,c),(d,e)],
            [(a,f),(b,d),(c,e)],
            [(a,f),(b,e),(c,d)],
        ]
        for matching in matchings:
            (p1, p2), (p3, p4), (p5, p6) = matching
            # The 8 transversal triples
            trans_triples = [
                (p1, p3, p5), (p1, p3, p6), (p1, p4, p5), (p1, p4, p6),
                (p2, p3, p5), (p2, p3, p6), (p2, p4, p5), (p2, p4, p6),
            ]
            # Sort each triple and convert to index
            tidxs = [triple_to_idx[tuple(sorted(t))] for t in trans_triples]
            # All pairs in this K(2,2,2) conflict
            for i in range(8):
                for j in range(i+1, 8):
                    conflict[tidxs[i]].add(tidxs[j])
                    conflict[tidxs[j]].add(tidxs[i])
    
    if time.time() > deadline - 30:
        # Fallback: all triples through point 0
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    # Greedy construction with randomization
    def greedy_randomize(order):
        selected = set()
        for idx in order:
            # Check if adding idx conflicts with any selected
            ok = True
            for sel in selected:
                if sel in conflict[idx]:
                    ok = False
                    break
            if ok:
                selected.add(idx)
        return selected
    
    # Initial greedy
    indices = list(range(num_triples))
    random.shuffle(indices)
    best_set = greedy_randomize(indices)
    best_size = len(best_set)
    
    # Local search: try to add/remove to improve
    def local_search(selected_set, max_iters=500):
        selected = set(selected_set)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to add an unselected triple
            unselected = [i for i in range(num_triples) if i not in selected]
            random.shuffle(unselected)
            for idx in unselected:
                ok = True
                for sel in selected:
                    if sel in conflict[idx]:
                        ok = False
                        break
                if ok:
                    selected.add(idx)
                    improved = True
                    break
            # Try swaps: remove one, add two
            if not improved:
                sel_list = list(selected)
                random.shuffle(sel_list)
                for rem in sel_list:
                    # Try to add two triples that conflict with rem but not each other
                    candidates = [i for i in range(num_triples) if i not in selected and rem in conflict[i]]
                    random.shuffle(candidates)
                    for c1 in candidates[:20]:
                        for c2 in candidates[:20]:
                            if c1 == c2:
                                continue
                            if c2 in conflict[c1]:
                                continue
                            # Check c1 and c2 don't conflict with other selected (except rem)
                            ok1 = all(not (s in conflict[c1]) for s in selected if s != rem)
                            if not ok1:
                                continue
                            ok2 = all(not (s in conflict[c2]) for s in selected if s != rem)
                            if not ok2:
                                continue
                            # Do the swap
                            selected.discard(rem)
                            selected.add(c1)
                            selected.add(c2)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
        return selected
    
    # Run multiple rounds of local search
    current_set = set(best_set)
    for round_num in range(20):
        if time.time() > deadline - 20:
            break
        
        # Local search
        new_set = local_search(current_set, max_iters=300)
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = new_set
            current_set = new_set
        
        # Perturbation: remove some random elements and re-greedy
        if time.time() > deadline - 15:
            break
        perturbed = set(current_set)
        num_remove = max(1, len(perturbed) // 10)
        to_remove = random.sample(list(perturbed), num_remove)
        for r in to_remove:
            perturbed.discard(r)
        # Re-add greedily
        unselected = [i for i in range(num_triples) if i not in perturbed]
        random.shuffle(unselected)
        for idx in unselected:
            ok = all(not (s in conflict[idx]) for s in perturbed)
            if ok:
                perturbed.add(idx)
        
        if len(perturbed) > best_size:
            best_size = len(perturbed)
            best_set = perturbed
            current_set = perturbed
    
    # Convert back to list of triples
    result = [list(all_triples[i]) for i in sorted(best_set)]
    return result

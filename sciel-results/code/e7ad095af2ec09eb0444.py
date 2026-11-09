"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Precompute conflict matrix: conflict[a][b] = True if lcm(a,b) <= 256
    N = 256
    size = N - 1  # numbers 2..256, index i = number i+2
    conflict = [[False] * size for _ in range(size)]
    
    for i in range(size):
        a = i + 2
        for j in range(i + 1, size):
            b = j + 2
            g = gcd(a, b)
            if a * b // g <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Precompute weights as Fractions for exact comparison
    weights = [Fraction(1, i + 2) for i in range(size)]
    
    # Precompute conflict sets for faster lookups
    conflict_sets = [set() for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if conflict[i][j]:
                conflict_sets[i].add(j)
    
    def score_of(selected_indices):
        s = Fraction(0)
        for idx in selected_indices:
            s += weights[idx]
        return s
    
    def can_add(selected_set, idx):
        """Check if idx can be added to selected_set."""
        return not conflict_sets[idx] & selected_set
    
    def greedy(order):
        """Greedy: add elements in given order if compatible."""
        selected = set()
        for idx in order:
            if can_add(selected, idx):
                selected.add(idx)
        return selected
    
    def local_search(selected_set, max_iters=20000):
        """1-swap and 2-swap local search with optimized 2-swap."""
        selected = set(selected_set)
        current_score = score_of(selected)
        best_score = current_score
        best_set = set(selected)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # 1-swap: remove one, add another
            for rem in list(selected):
                rem_weight = weights[rem]
                # Candidates that conflict with rem but not with (selected - {rem})
                others = selected - {rem}
                for add in range(size):
                    if add in selected:
                        continue
                    # add must not conflict with any element in others
                    if conflict_sets[add] & others:
                        continue
                    new_score = current_score - rem_weight + weights[add]
                    if new_score > current_score:
                        current_score = new_score
                        selected = others | {add}
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # 2-swap: remove two, add two
                sel_list = list(selected)
                n_sel = len(sel_list)
                found = False
                for i in range(n_sel):
                    if time.time() > deadline:
                        break
                    rem1 = sel_list[i]
                    for j in range(i + 1, n_sel):
                        rem2 = sel_list[j]
                        base = selected - {rem1, rem2}
                        # Find candidates that can be added to base
                        # and don't conflict with each other
                        candidates = []
                        for add in range(size):
                            if add in selected:
                                continue
                            if conflict_sets[add] & base:
                                continue
                            candidates.append(add)
                        
                        # Try pairs from candidates
                        for ci in range(len(candidates)):
                            if time.time() > deadline:
                                break
                            add1 = candidates[ci]
                            for cj in range(ci + 1, len(candidates)):
                                add2 = candidates[cj]
                                if conflict[add1][add2]:
                                    continue
                                new_score = (current_score 
                                            - weights[rem1] - weights[rem2] 
                                            + weights[add1] + weights[add2])
                                if new_score > current_score:
                                    current_score = new_score
                                    selected = base | {add1, add2}
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_set = set(selected)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not improved:
                break
        
        return best_set
    
    # Generate diverse starting points
    start_points = []
    
    # Order 1: ascending
    start_points.append(list(range(size)))
    
    # Order 2: descending
    start_points.append(list(range(size - 1, -1, -1)))
    
    # Order 3: by weight descending (small numbers first - highest weight)
    order3 = sorted(range(size), key=lambda i: -weights[i])
    start_points.append(order3)
    
    # Order 4: by number of conflicts ascending (least constrained first)
    conf_counts = [len(conflict_sets[i]) for i in range(size)]
    order4 = sorted(range(size), key=lambda i: conf_counts[i])
    start_points.append(order4)
    
    # Order 5: by weight/conflict ratio
    order5 = sorted(range(size), key=lambda i: -weights[i] / (conf_counts[i] + 1))
    start_points.append(order5)
    
    # Order 6: random seed 42
    import random
    rng = random.Random(42)
    order6 = list(range(size))
    rng.shuffle(order6)
    start_points.append(order6)
    
    # Order 7: random seed 123
    rng2 = random.Random(123)
    order7 = list(range(size))
    rng2.shuffle(order7)
    start_points.append(order7)
    
    # Order 8: random seed 999
    rng3 = random.Random(999)
    order8 = list(range(size))
    rng3.shuffle(order8)
    start_points.append(order8)
    
    # Order 9: weight descending, tie-break by conflict count ascending
    order9 = sorted(range(size), key=lambda i: (-weights[i], conf_counts[i]))
    start_points.append(order9)
    
    # Order 10: conflict count ascending, tie-break by weight descending
    order10 = sorted(range(size), key=lambda i: (conf_counts[i], -weights[i]))
    start_points.append(order10)
    
    # Run greedy on each starting order, then local search
    best_overall = set()
    best_overall_score = Fraction(0)
    
    for order in start_points:
        if time.time() > deadline:
            break
        gs = greedy(order)
        if not gs:
            continue
        ls = local_search(gs, max_iters=20000)
        sc = score_of(ls)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = ls
    
    # Convert back to numbers
    result = [i + 2 for i in best_overall]
    if not result:
        result = [256]  # fallback
    
    return sorted(result)
# EVOLVE-BLOCK-END

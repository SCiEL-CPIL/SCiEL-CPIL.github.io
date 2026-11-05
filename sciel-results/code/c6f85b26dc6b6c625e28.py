import time
import random
import numpy as np
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill

    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute conflict relationships
    conflict = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            g = gcd(vi, vj)
            l = vi * vj // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # 3. Precompute reciprocal values as floats for fast comparison
    reciprocals = [1.0 / candidates[i] for i in range(n)]
    
    def score_float(selected_set):
        """Compute reciprocal sum as float for fast comparison"""
        return sum(reciprocals[idx] for idx in selected_set)
    
    def greedy_select(order):
        """Greedy selection following given order"""
        selected = []
        sel_set = set()
        for idx in order:
            if not (sel_set & conflict[idx]):
                selected.append(idx)
                sel_set.add(idx)
        return selected
    
    def local_search(selected, max_iters=200, time_limit=None):
        """Local search: try additions, removals, and swaps using float scores"""
        selected_set = set(selected)
        current_score = score_float(selected_set)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            
            # Try adding each unselected element
            for idx in range(n):
                if idx not in selected_set:
                    if not (selected_set & conflict[idx]):
                        new_score = current_score + reciprocals[idx]
                        if new_score > current_score + 1e-15:
                            selected_set.add(idx)
                            current_score = new_score
                            improved = True
            
            # Try 1-for-1 swaps
            for remove_idx in list(selected_set):
                for add_idx in range(n):
                    if add_idx in selected_set:
                        continue
                    remaining = selected_set - {remove_idx}
                    if remaining & conflict[add_idx]:
                        continue
                    new_score = current_score - reciprocals[remove_idx] + reciprocals[add_idx]
                    if new_score > current_score + 1e-15:
                        selected_set.remove(remove_idx)
                        selected_set.add(add_idx)
                        current_score = new_score
                        improved = True
                        break
            
            # Try 2-for-2 swaps: remove two, add two
            sel_list = list(selected_set)
            k = len(sel_list)
            for i in range(k):
                if improved:
                    break
                for j in range(i + 1, k):
                    if improved:
                        break
                    r1, r2 = sel_list[i], sel_list[j]
                    remaining = selected_set - {r1, r2}
                    for a1 in range(n):
                        if improved:
                            break
                        if a1 in selected_set or a1 == r1 or a1 == r2:
                            continue
                        if remaining & conflict[a1]:
                            continue
                        for a2 in range(n):
                            if a2 in selected_set or a2 == r1 or a2 == r2 or a2 == a1:
                                continue
                            if (remaining | {a1}) & conflict[a2]:
                                continue
                            new_score = current_score - reciprocals[r1] - reciprocals[r2] + reciprocals[a1] + reciprocals[a2]
                            if new_score > current_score + 1e-15:
                                selected_set.discard(r1)
                                selected_set.discard(r2)
                                selected_set.add(a1)
                                selected_set.add(a2)
                                current_score = new_score
                                improved = True
                                break
        
        return list(selected_set), current_score
    
    def shake(selected_set, num_remove=3):
        """Random perturbation: remove random elements, then greedily re-add"""
        sel = list(selected_set)
        if len(sel) <= num_remove:
            return selected_set
        to_remove = random.sample(sel, num_remove)
        perturbed = selected_set - set(to_remove)
        # Greedily add back best compatible elements
        candidates_to_add = [i for i in range(n) if i not in perturbed]
        candidates_to_add.sort(key=lambda i: -reciprocals[i])
        for idx in candidates_to_add:
            if not (perturbed & conflict[idx]):
                perturbed.add(idx)
        return perturbed
    
    # 4. Multiple restarts with different greedy orders
    best_result = None
    best_score_float = 0.0
    
    random.seed(42)
    
    # Strategy 1: Sort by value ascending (smallest values first - highest reciprocal)
    order1 = sorted(range(n), key=lambda i: candidates[i])
    sel1 = greedy_select(order1)
    sel1, score1 = local_search(sel1, max_iters=100, time_limit=deadline - 5)
    if score1 > best_score_float:
        best_score_float = score1
        best_result = sel1
    
    # Strategy 2: Sort by value descending
    order2 = sorted(range(n), key=lambda i: -candidates[i])
    sel2 = greedy_select(order2)
    sel2, score2 = local_search(sel2, max_iters=100, time_limit=deadline - 5)
    if score2 > best_score_float:
        best_score_float = score2
        best_result = sel2
    
    # Strategy 3: Sort by number of conflicts (fewest conflicts first)
    order3 = sorted(range(n), key=lambda i: len(conflict[i]))
    sel3 = greedy_select(order3)
    sel3, score3 = local_search(sel3, max_iters=100, time_limit=deadline - 5)
    if score3 > best_score_float:
        best_score_float = score3
        best_result = sel3
    
    # Strategy 4: Sort by reciprocal / (conflict_count + 1)
    order4 = sorted(range(n), key=lambda i: -reciprocals[i] / (len(conflict[i]) + 1))
    sel4 = greedy_select(order4)
    sel4, score4 = local_search(sel4, max_iters=100, time_limit=deadline - 5)
    if score4 > best_score_float:
        best_score_float = score4
        best_result = sel4
    
    # Strategy 5: Random shuffles with shake-and-refine
    num_restarts = 0
    while time.time() - start_time < deadline - 3:
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_select(order)
        sel, sc = local_search(sel, max_iters=80, time_limit=deadline - 3)
        if sc > best_score_float:
            best_score_float = sc
            best_result = sel
        num_restarts += 1
        
        # Periodically try shake on current best
        if num_restarts % 5 == 0 and best_result is not None:
            perturbed = shake(set(best_result), num_remove=random.randint(2, 4))
            perturbed_list = list(perturbed)
            perturbed_list, sc2 = local_search(perturbed_list, max_iters=100, time_limit=deadline - 3)
            if sc2 > best_score_float:
                best_score_float = sc2
                best_result = perturbed_list
    
    # Final intensive local search on best result
    if time.time() - start_time < deadline - 1:
        best_result, best_score_float = local_search(best_result, max_iters=500, time_limit=deadline - 0.5)
    
    # Convert indices back to values
    result = [candidates[i] for i in best_result]
    
    # Verify validity
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result:
        # Fallback: simple greedy
        selected = []
        for v in sorted(candidates):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)
        return selected
    
    return result

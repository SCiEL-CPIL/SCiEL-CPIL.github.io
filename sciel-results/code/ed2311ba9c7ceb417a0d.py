"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 50.0  # 50s safety margin within 60s limit
    
    # Denominators are 2..48, indices 0..46 correspond to d=2..48
    n = 47
    denominators = list(range(2, 49))
    
    # Precompute Fraction values
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    selected = []
    total = Fraction(0, 1)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    best_set = set(selected)
    best_deficit = 1 - total  # positive deficit
    
    # Local search: try to improve by swapping or adding/removing elements
    # We want to minimize deficit = 1 - sum, i.e., maximize sum subject to sum < 1
    
    def get_deficit(s):
        t = Fraction(0, 1)
        for i in s:
            t += reciprocals[i]
        return 1 - t
    
    def is_valid(s):
        t = Fraction(0, 1)
        for i in s:
            t += reciprocals[i]
        return t < 1
    
    # Local improvement: try all single additions, removals, and swaps
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_set = set(best_set)
        
        # Try adding one element
        for i in range(n):
            if i not in current_set:
                new_set = current_set | {i}
                t = Fraction(0, 1)
                for j in new_set:
                    t += reciprocals[j]
                if t < 1:
                    new_deficit = 1 - t
                    if new_deficit < best_deficit:
                        best_set = new_set
                        best_deficit = new_deficit
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try removing one element (to allow other additions later)
        for i in list(best_set):
            new_set = best_set - {i}
            t = Fraction(0, 1)
            for j in new_set:
                t += reciprocals[j]
            if t < 1:
                new_deficit = 1 - t
                if new_deficit < best_deficit:
                    best_set = new_set
                    best_deficit = new_deficit
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping: remove one, add one
        for i in list(best_set):
            for j in range(n):
                if j not in best_set:
                    new_set = (best_set - {i}) | {j}
                    t = Fraction(0, 1)
                    for k in new_set:
                        t += reciprocals[k]
                    if t < 1:
                        new_deficit = 1 - t
                        if new_deficit < best_deficit:
                            best_set = new_set
                            best_deficit = new_deficit
                            improved = True
                            break
            if improved:
                break
        
        if time.time() > deadline - 10.0:
            break
    
    # More aggressive local search with random restarts if time permits
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Random perturbation of current best
            current_set = set(best_set)
            # Randomly remove 1-3 elements
            to_remove = list(current_set)
            rng.shuffle(to_remove)
            for _ in range(min(3, len(to_remove))):
                current_set.discard(to_remove.pop())
            
            # Try to greedily add back
            for i in range(n):
                if i not in current_set:
                    t = Fraction(0, 1)
                    for j in current_set:
                        t += reciprocals[j]
                    if t + reciprocals[i] < 1:
                        current_set.add(i)
            
            # Local search from this perturbed state
            local_improved = True
            while local_improved and time.time() < deadline - 5.0:
                local_improved = False
                # Try additions
                for i in range(n):
                    if i not in current_set:
                        t = Fraction(0, 1)
                        for j in current_set:
                            t += reciprocals[j]
                        if t + reciprocals[i] < 1:
                            new_deficit = 1 - (t + reciprocals[i])
                            if new_deficit < best_deficit:
                                current_set.add(i)
                                best_set = set(current_set)
                                best_deficit = new_deficit
                                local_improved = True
                                break
                # Try swaps
                for i in list(current_set):
                    for j in range(n):
                        if j not in current_set:
                            t = Fraction(0, 1)
                            for k in current_set:
                                if k != i:
                                    t += reciprocals[k]
                            t += reciprocals[j]
                            if t < 1:
                                new_deficit = 1 - t
                                if new_deficit < best_deficit:
                                    current_set = (current_set - {i}) | {j}
                                    best_set = set(current_set)
                                    best_deficit = new_deficit
                                    local_improved = True
                                    break
                    if local_improved:
                        break
    
    return list(best_set)
# EVOLVE-BLOCK-END

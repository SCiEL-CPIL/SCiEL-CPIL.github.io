import math
import time
import random
from itertools import combinations

def construct():
    N = 512
    deadline = time.time() + 145.0
    
    # Precompute values: value[a] = 1/(N - a) for a in 1..511
    value = [0.0] * N
    for a in range(1, N):
        value[a] = 1.0 / (N - a)
    
    # Precompute gcd matrix for all pairs
    # gcd_ok[a][b] = True if gcd(a,b) == 1
    # Actually, let's just use math.gcd since it's fast enough
    
    def score(arr):
        return sum(value[a] for a in arr)
    
    def is_coprime_set(chosen, a):
        for b in chosen:
            if math.gcd(a, b) > 1:
                return False
        return True
    
    # Greedy initialization: pick in order of decreasing value (smallest d = N-a first)
    # So a = N-1, N-2, ..., 1 -> but we want largest a first for highest value
    # value[a] = 1/(N-a), so larger a gives larger value
    # So sort by a descending: 511, 510, ..., 1
    chosen = []
    for a in range(N - 1, 0, -1):
        if is_coprime_set(chosen, a):
            chosen.append(a)
    
    best_score = score(chosen)
    best_set = set(chosen)
    best_list = list(chosen)
    
    # Local search: 1-swap and 2-swap
    # For 1-swap: try removing a from chosen and adding b not in chosen
    # For 2-swap: try removing a1,a2 from chosen and adding b1,b2 not in chosen
    
    def try_1_swap(chosen, best_score, best_set, best_list):
        chosen_set = set(chosen)
        improved = False
        for a in list(chosen):
            for b in range(N - 1, 0, -1):
                if b in chosen_set:
                    continue
                # Check if b is coprime with all other elements in chosen (except a)
                ok = True
                for c in chosen:
                    if c == a:
                        continue
                    if math.gcd(b, c) > 1:
                        ok = False
                        break
                if not ok:
                    continue
                new_score = best_score - value[a] + value[b]
                if new_score > best_score + 1e-12:
                    new_chosen = [c for c in chosen if c != a]
                    new_chosen.append(b)
                    best_score = new_score
                    best_set = set(new_chosen)
                    best_list = list(new_chosen)
                    improved = True
                    break
            if time.time() > deadline - 5.0:
                break
        return best_score, best_set, best_list, improved
    
    def try_2_swap(chosen, best_score, best_set, best_list):
        chosen_set = set(chosen)
        improved = False
        n = len(chosen)
        for i in range(n):
            a1 = chosen[i]
            for j in range(i + 1, n):
                a2 = chosen[j]
                # Try adding b1, b2 not in chosen
                # b1 > b2 for ordering
                for b1 in range(N - 1, 0, -1):
                    if b1 in chosen_set:
                        continue
                    for b2 in range(b1 - 1, 0, -1):
                        if b2 in chosen_set:
                            continue
                        # Check b1 and b2 are coprime with each other
                        if math.gcd(b1, b2) > 1:
                            continue
                        # Check b1 and b2 are coprime with all elements in chosen except a1, a2
                        ok = True
                        for c in chosen:
                            if c == a1 or c == a2:
                                continue
                            if math.gcd(b1, c) > 1 or math.gcd(b2, c) > 1:
                                ok = False
                                break
                        if not ok:
                            continue
                        new_score = best_score - value[a1] - value[a2] + value[b1] + value[b2]
                        if new_score > best_score + 1e-12:
                            new_chosen = [c for c in chosen if c != a1 and c != a2]
                            new_chosen.append(b1)
                            new_chosen.append(b2)
                            best_score = new_score
                            best_set = set(new_chosen)
                            best_list = list(new_chosen)
                            improved = True
                            break
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        return best_score, best_set, best_list, improved
    
    # Run local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_score, best_set, best_list, imp1 = try_1_swap(best_list, best_score, best_set, best_list)
        if imp1:
            improved = True
            continue
        best_score, best_set, best_list, imp2 = try_2_swap(best_list, best_score, best_set, best_list)
        if imp2:
            improved = True
    
    return best_list

"""Combined program: precompute conflict pairs, fast greedy with multiple orderings,
and multi-level local search (add, 1-for-2, 2-for-3, 1-for-1 plateau hopping)."""
import time
import random
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _precompute_conflicts(n=128):
    """Precompute bad_pairs_for[x]: set of (u, v) with u < v such that {x, u, v} is a bad triple."""
    bad_pairs_for = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    return bad_pairs_for

def _can_add(x, current_set, bad_pairs_for):
    """Check if x can be added to current_set without forming a bad triple."""
    for (u, v) in bad_pairs_for[x]:
        if u in current_set and v in current_set:
            return False
    return True

def _greedy(order, bad_pairs_for):
    """Greedy: add numbers in given order if no bad triple is formed."""
    selected = set()
    for x in order:
        if _can_add(x, selected, bad_pairs_for):
            selected.add(x)
    return selected

def _local_search(S, bad_pairs_for, deadline):
    """Multi-level local search: try to add, 1-for-2 swap, 2-for-3 swap, 1-for-1 plateau hop."""
    S = set(S)
    n = 128
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. Try to add any element
        for x in range(1, n + 1):
            if x not in S and _can_add(x, S, bad_pairs_for):
                S.add(x)
                improved = True
                break
        
        if not improved:
            # 2. Try 1-for-2 swap: remove x, add two elements y, z
            found_swap = False
            for x in list(S):
                S_trial = S - {x}
                added = []
                for y in range(1, n + 1):
                    if y not in S_trial and _can_add(y, S_trial, bad_pairs_for):
                        S_trial.add(y)
                        added.append(y)
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    S = S_trial
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                continue
        
        if not improved:
            # 3. Try 2-for-3 swap: remove x, y, add three elements
            found_swap = False
            s_list = list(S)
            for i in range(len(s_list)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(s_list)):
                    S_trial = S - {s_list[i], s_list[j]}
                    added = []
                    for y in range(1, n + 1):
                        if y not in S_trial and _can_add(y, S_trial, bad_pairs_for):
                            S_trial.add(y)
                            added.append(y)
                            if len(added) == 3:
                                break
                    if len(added) == 3:
                        S = S_trial
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                continue
        
        if not improved:
            # 4. Try 1-for-1 plateau hopping to escape local minima
            found_swap = False
            for _ in range(10):
                x = random.choice(list(S))
                S_trial = S - {x}
                candidates = [y for y in range(1, n + 1) if y not in S_trial and _can_add(y, S_trial, bad_pairs_for)]
                if candidates:
                    y = random.choice(candidates)
                    S_trial.add(y)
                    S = S_trial
                    found_swap = True
                    break
            # Plateau hopping doesn't increase size, so don't set improved = True
            # But we break out to avoid infinite loop; the while condition will exit
            if found_swap:
                # Continue searching but don't mark as improved
                pass
    
    return S

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Precompute conflict structures
    bad_pairs_for = _precompute_conflicts()
    
    if time.time() > deadline:
        return [1]
    
    best_set = set()
    best_size = 0
    
    # Precompute conflict degrees for ordering
    degrees = [len(bad_pairs_for[k]) for k in range(129)]
    
    # Run multiple randomized greedy passes with different orderings
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 15:  # Leave time for final local search
            break
        
        # Choose ordering strategy
        if i % 3 == 0:
            # Random order
            order = list(range(1, 129))
            random.shuffle(order)
        elif i % 3 == 1:
            # Sort by conflict degree with random noise
            order = list(range(1, 129))
            order.sort(key=lambda x: degrees[x] + random.random())
        else:
            # Deterministic: natural or reverse
            if i % 6 == 2:
                order = list(range(1, 129))
            else:
                order = list(range(128, 0, -1))
        
        S = _greedy(order, bad_pairs_for)
        S = _local_search(S, bad_pairs_for, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final polish on best set
    if time.time() < deadline - 2:
        best_set = _local_search(best_set, bad_pairs_for, deadline)
    
    return list(best_set)

import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute bad pairs
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) != ab:
                    continue
                if _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    def can_add(x, current_set):
        for (a, b) in bad_pairs_for[x]:
            if a in current_set and b in current_set:
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(S):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            S = set(S)
            
            # 1. Add any possible
            for x in range(1, n + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # 2. 1-for-1 swap
                found_swap = False
                for x in list(S):
                    S_trial = S - {x}
                    for y in range(1, n + 1):
                        if y not in S_trial and can_add(y, S_trial):
                            S_trial.add(y)
                            if len(S_trial) > len(S): # Actually size is same, so this check is redundant if we just want to swap
                                # But we need to ensure we actually added y
                                S = S_trial
                                improved = True
                                found_swap = True
                                break
                    if found_swap:
                        break
                if found_swap:
                    continue
            
            if not improved:
                # 3. 1-for-2 swap
                found_swap = False
                for x in list(S):
                    S_trial = S - {x}
                    added = 0
                    for y in range(1, n + 1):
                        if y not in S_trial and can_add(y, S_trial):
                            S_trial.add(y)
                            added += 1
                            if added >= 2:
                                break
                    if len(S_trial) > len(S):
                        S = S_trial
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue
            
            if not improved:
                # 4. 2-for-3 swap
                found_swap = False
                # Sample a subset of pairs to avoid O(N^4) if needed, but N=128 is small
                # Pairs in S: ~100*100/2 = 5000
                # For each pair, try to add 3
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() > deadline - 2: break
                    for j in range(i + 1, len(s_list)):
                        S_trial = S - {s_list[i], s_list[j]}
                        added = 0
                        for y in range(1, n + 1):
                            if y not in S_trial and can_add(y, S_trial):
                                S_trial.add(y)
                                added += 1
                                if added >= 3:
                                    break
                        if len(S_trial) > len(S):
                            S = S_trial
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue
                    
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple randomized greedy passes
    num_restarts = 100 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final local search on best
            break
            
        # Randomized order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Also mix in some deterministic biases
        if i % 3 == 0:
             order = sorted(order, key=lambda x: random.random())
        elif i % 3 == 1:
             # Sort by number of bad pairs (conflict degree)
             degrees = [len(bad_pairs_for[k]) for k in range(n+1)]
             # Add random noise to degree
             order = sorted(order, key=lambda x: degrees[x] + random.random())
        else:
             # Just random
             order = list(range(1, n+1))
             random.shuffle(order)
        S = greedy(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on best_set
    best_set = local_search(best_set)
    
    return list(best_set)

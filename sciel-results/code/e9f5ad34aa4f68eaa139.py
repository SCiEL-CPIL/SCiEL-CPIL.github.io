import time
import math
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def _find_components(N, triples, elem_triples):
    """Find connected components in the conflict hypergraph."""
    adj = defaultdict(set)
    for (a, b, c) in triples:
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    
    visited = set()
    components = []
    for node in range(1, N + 1):
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    comp.add(v)
                    for neighbor in adj[v]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(comp)
    return components

def _solve_component_exhaustive(comp, triples, N):
    """
    For a small component, try to find the maximum independent set.
    Use branch and bound with bitmasks if component size <= 20.
    """
    if len(comp) <= 1:
        return set(comp)
    
    comp_list = sorted(comp)
    comp_set = set(comp_list)
    
    # Filter triples that are entirely within this component
    comp_triples = []
    for (a, b, c) in triples:
        if a in comp_set and b in comp_set and c in comp_set:
            comp_triples.append((a, b, c))
    
    if not comp_triples:
        return set(comp_list)
    
    # If component is small enough, do exhaustive search with pruning
    if len(comp) <= 15:
        # Map elements to indices
        elem_to_idx = {v: i for i, v in enumerate(comp_list)}
        n = len(comp_list)
        
        # Build conflict mask for each element (bitmask of elements that form a triple with it)
        # Actually, we need: for each element, which other elements can't be with it
        # A triple (a,b,c) means we can't have all three. So it's a 3-uniform hypergraph.
        # For small n, we can enumerate subsets.
        
        # Use recursive backtracking
        best = set()
        
        def backtrack(idx, current, remaining):
            nonlocal best
            if idx == n:
                if len(current) > len(best):
                    best = set(current)
                return
            
            # Pruning: even if we add all remaining, can't beat best
            if len(current) + (n - idx) <= len(best):
                return
            
            # Try not including comp_list[idx]
            backtrack(idx + 1, current, remaining - 1)
            
            # Try including comp_list[idx]
            v = comp_list[idx]
            # Check if adding v creates a complete triple
            valid = True
            for (a, b, c) in comp_triples:
                if v in (a, b, c):
                    other = [x for x in (a, b, c) if x != v]
                    if all(x in current for x in other):
                        valid = False
                        break
            if valid:
                current.append(v)
                backtrack(idx + 1, current, remaining - 1)
                current.pop()
        
        backtrack(0, [], n)
        return best
    else:
        # For larger components, use greedy + local search
        return _solve_component_sa(comp, comp_triples, N, time_limit=2.0)

def _solve_component_sa(comp, comp_triples, N, time_limit=2.0):
    """SA for a single component."""
    start = time.time()
    deadline = start + time_limit
    
    comp_set = set(comp)
    # Initialize: try all elements, then remove conflicts greedily
    S = set(comp)
    counts = {t: 3 for t in comp_triples}
    
    # Greedy removal
    while True:
        active = [t for t in comp_triples if counts[t] == 3]
        if not active:
            break
        # Find element in most active triples
        elem_counts = defaultdict(int)
        for t in active:
            for x in t:
                if x in S:
                    elem_counts[x] += 1
        if not elem_counts:
            break
        max_count = max(elem_counts.values())
        candidates = [x for x, c in elem_counts.items() if c == max_count]
        x_remove = random.choice(candidates)
        S.remove(x_remove)
        for t in comp_triples:
            if x_remove in t:
                counts[t] -= 1
    
    best_set = set(S)
    best_size = len(S)
    
    # SA
    T = 5.0
    alpha = 0.995
    
    while time.time() < deadline:
        if T < 0.01:
            break
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            move = random.random()
            if move < 0.4:
                # Add
                not_in = [x for x in comp if x not in S]
                if not not_in:
                    continue
                x = random.choice(not_in)
                delta_active = 0
                for t in comp_triples:
                    if x in t and counts[t] == 2:
                        delta_active += 1
                delta_cost = -1 + 50 * delta_active
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    S.add(x)
                    for t in comp_triples:
                        if x in t:
                            counts[t] += 1
                    if all(counts[t] < 3 for t in comp_triples):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
            elif move < 0.7:
                # Remove
                if not S:
                    continue
                x = random.choice(list(S))
                delta_active = 0
                for t in comp_triples:
                    if x in t and counts[t] == 3:
                        delta_active -= 1
                delta_cost = 1 + 50 * delta_active
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    S.remove(x)
                    for t in comp_triples:
                        if x in t:
                            counts[t] -= 1
                    if all(counts[t] < 3 for t in comp_triples):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
            else:
                # Swap
                if not S:
                    continue
                not_in = [x for x in comp if x not in S]
                if not not_in:
                    continue
                x_out = random.choice(list(S))
                x_in = random.choice(not_in)
                if x_out == x_in:
                    continue
                # Calculate delta
                delta_active = 0
                affected = set()
                for t in comp_triples:
                    if x_out in t or x_in in t:
                        affected.add(t)
                for t in affected:
                    old_c = counts[t]
                    new_c = old_c
                    if x_out in t:
                        new_c -= 1
                    if x_in in t:
                        new_c += 1
                    if old_c == 3 and new_c != 3:
                        delta_active -= 1
                    elif old_c != 3 and new_c == 3:
                        delta_active += 1
                delta_cost = 50 * delta_active
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    S.remove(x_out)
                    S.add(x_in)
                    for t in affected:
                        if x_out in t:
                            counts[t] -= 1
                        if x_in in t:
                            counts[t] += 1
                    if all(counts[t] < 3 for t in comp_triples):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
        
        T *= alpha
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    # Find connected components
    components = _find_components(N, triples, elem_triples)
    
    # Solve each component
    best_set = set()
    remaining_time = deadline - time.time()
    
    for i, comp in enumerate(components):
        if time.time() > deadline - 3.0:
            break
        time_per_comp = max(0.5, remaining_time / max(1, len(components) - i))
        
        if len(comp) <= 15:
            sol = _solve_component_exhaustive(comp, triples, N)
        else:
            sol = _solve_component_sa(comp, [t for t in triples if all(x in comp for x in t)], N, time_limit=time_per_comp)
        
        best_set.update(sol)
    
    # If we used up time, do a global polish
    if time.time() < deadline - 2.0:
        # Global SA polish on the combined set
        S = set(best_set)
        counts = [0] * T
        for idx, (a, b, c) in enumerate(triples):
            cnt = 0
            if a in S: cnt += 1
            if b in S: cnt += 1
            if c in S: cnt += 1
            counts[idx] = cnt
        
        active_count = sum(1 for c in counts if c == 3)
        best_size = len(S)
        best_set_polish = set(S)
        
        T_sa = 5.0
        alpha = 0.999
        in_S = [False] * (N + 1)
        for x in S:
            in_S[x] = True
        
        while time.time() < deadline - 1.0 and T_sa > 0.01:
            for _ in range(500):
                if time.time() > deadline - 1.0:
                    break
                
                move = random.random()
                if move < 0.35:
                    # Add
                    not_in = [x for x in range(1, N+1) if not in_S[x]]
                    if not not_in:
                        continue
                    x = random.choice(not_in)
                    delta_active = 0
                    for t_idx in elem_triples[x]:
                        if counts[t_idx] == 2:
                            delta_active += 1
                    delta_cost = -1 + 100 * delta_active
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T_sa):
                        S.add(x)
                        in_S[x] = True
                        for t_idx in elem_triples[x]:
                            counts[t_idx] += 1
                        active_count += delta_active
                        if active_count == 0 and len(S) > best_size:
                            best_size = len(S)
                            best_set_polish = set(S)
                elif move < 0.65:
                    # Remove
                    if not S:
                        continue
                    x = random.choice(list(S))
                    delta_active = 0
                    for t_idx in elem_triples[x]:
                        if counts[t_idx] == 3:
                            delta_active -= 1
                    delta_cost = 1 + 100 * delta_active
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T_sa):
                        S.remove(x)
                        in_S[x] = False
                        for t_idx in elem_triples[x]:
                            counts[t_idx] -= 1
                        active_count += delta_active
                        if active_count == 0 and len(S) > best_size:
                            best_size = len(S)
                            best_set_polish = set(S)
                else:
                    # Swap
                    if not S:
                        continue
                    not_in = [x for x in range(1, N+1) if not in_S[x]]
                    if not not_in:
                        continue
                    x_out = random.choice(list(S))
                    x_in = random.choice(not_in)
                    if x_out == x_in:
                        continue
                    
                    # Calculate delta using affected triples
                    affected = set(elem_triples[x_out]) | set(elem_triples[x_in])
                    delta_active = 0
                    for t_idx in affected:
                        old_c = counts[t_idx]
                        new_c = old_c
                        a, b, c = triples[t_idx]
                        if x_out == a or x_out == b or x_out == c:
                            new_c -= 1
                        if x_in == a or x_in == b or x_in == c:
                            new_c += 1
                        if old_c == 3 and new_c != 3:
                            delta_active -= 1
                        elif old_c != 3 and new_c == 3:
                            delta_active += 1
                    
                    delta_cost = 100 * delta_active
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T_sa):
                        S.remove(x_out)
                        in_S[x_out] = False
                        S.add(x_in)
                        in_S[x_in] = True
                        for t_idx in affected:
                            a, b, c = triples[t_idx]
                            if x_out == a or x_out == b or x_out == c:
                                counts[t_idx] -= 1
                            if x_in == a or x_in == b or x_in == c:
                                counts[t_idx] += 1
                        active_count += delta_active
                        if active_count == 0 and len(S) > best_size:
                            best_size = len(S)
                            best_set_polish = set(S)
            
            T_sa *= alpha
        
        best_set = best_set_polish
    
    # Final validity check and repair
    if not best_set:
        best_set = set(range(90, 181))
        
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c)
    
    return list(best_set)

# EVOLVE-BLOCK-END

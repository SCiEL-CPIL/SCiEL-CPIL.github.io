import time
import sys
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def hungarian(cost):
    n_local = len(cost)
    INF = float('inf')
    u = [0] * (n_local + 1)
    v = [0] * (n_local + 1)
    p = [0] * (n_local + 1)
    way = [0] * (n_local + 1)
    for i in range(1, n_local + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n_local + 1)
        used = [False] * (n_local + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n_local + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n_local + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n_local
    for j in range(1, n_local + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment

def find_cycles(assignment):
    n_local = len(assignment)
    visited = [False] * n_local
    cycles = []
    for start in range(n_local):
        if visited[start]:
            continue
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = assignment[cur]
        cycles.append(cycle)
    return cycles

def merge_two_cycles(c1, c2):
    """Merge two cycles by trying all edge swaps. Returns (merged_cycle, new_score_delta) or None."""
    k = len(c1)
    l = len(c2)
    best = None
    best_score = float('inf')
    for ai in range(k):
        for bj in range(l):
            u1 = c1[ai]
            v1 = c1[(ai + 1) % k]
            u2 = c2[bj]
            v2 = c2[(bj + 1) % l]
            # Remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
            # Merged: v1, ..., u1, v2, ..., u2
            ci_part = [c1[(ai + 1 + t) % k] for t in range(k)]
            cj_part = [c2[(bj + 1 + t) % l] for t in range(l)]
            merged = ci_part + cj_part
            score = 0
            for idx in range(len(merged)):
                a = merged[idx]
                b = merged[(idx + 1) % len(merged)]
                ar = edge_arrival(a, b) + 1
                if ar > score:
                    score = ar
            if score < best_score:
                best_score = score
                best = merged
    return best

def repair_cycle_cover(cycles, deadline):
    """Repeatedly merge pairs of cycles using best edge swap until one cycle remains."""
    if len(cycles) == 1:
        return cycles[0]
    
    while len(cycles) > 1:
        if time.time() > deadline:
            return None
        best_pair = None
        best_merged = None
        best_score = float('inf')
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                merged = merge_two_cycles(cycles[i], cycles[j])
                if merged is not None:
                    sc = cycle_score(merged)
                    if sc < best_score:
                        best_score = sc
                        best_pair = (i, j)
                        best_merged = merged
        if best_pair is None:
            return None
        i, j = best_pair
        new_cycles = []
        for idx_c, c in enumerate(cycles):
            if idx_c != i and idx_c != j:
                new_cycles.append(c)
        new_cycles.append(best_merged)
        cycles = new_cycles
    
    return cycles[0]

def two_opt_improve(cycle, deadline):
    """Apply 2-opt moves to reduce the bottleneck score of a Hamiltonian cycle."""
    L = len(cycle)
    current_score = cycle_score(cycle)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(L):
            if time.time() > deadline:
                return cycle
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue
                # 2-opt: reverse segment from i+1 to j
                a = cycle[i]
                b = cycle[(i + 1) % L]
                c = cycle[j]
                d = cycle[(j + 1) % L]
                # New edges: (a,c) and (b,d) replace (a,b) and (c,d)
                new_ar1 = edge_arrival(a, c) + 1
                new_ar2 = edge_arrival(b, d) + 1
                old_ar1 = edge_arrival(a, b) + 1
                old_ar2 = edge_arrival(c, d) + 1
                # Compute new max
                new_max = max(new_ar1, new_ar2)
                # Check if this reduces the bottleneck
                # The bottleneck is the max over all edges
                # If either old edge was the unique max and new edges are smaller, we improve
                if new_max < current_score:
                    # Verify by computing full score (expensive but safe)
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    ns = cycle_score(new_cycle)
                    if ns < current_score:
                        cycle = new_cycle
                        current_score = ns
                        improved = True
                        break
            if improved:
                break
    return cycle

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    best_cycle = None
    best_score = m + 1
    
    # Strategy: Use Hungarian with different cost formulations, then repair
    # Cost formulation 1: raw arrival index (minimize total -> good cycle cover)
    # Cost formulation 2: penalize high-arrival edges heavily (bottleneck-oriented)
    
    rng = random.Random(42)
    
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Build cost matrix
        # For bottleneck-oriented: use cost = arrival^2 or arrival^3 to push down max
        if attempt == 0:
            # Raw arrival
            power = 1
            perturb = 0.0
        elif attempt == 1:
            power = 2
            perturb = 0.0
        elif attempt == 2:
            power = 3
            perturb = 0.0
        elif attempt == 3:
            power = 4
            perturb = 0.0
        elif attempt < 10:
            power = 2
            perturb = rng.uniform(0, 5)
        else:
            power = 3
            perturb = rng.uniform(0, 10)
        
        INF = float('inf')
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ar = edge_arrival(i, j)
                cost[i][j] = ar ** power + perturb * rng.random()
        
        # Check feasibility
        feasible = True
        for i in range(n):
            row_ok = any(cost[i][j] < INF for j in range(n))
            col_ok = any(cost[j][i] < INF for j in range(n))
            if not row_ok or not col_ok:
                feasible = False
                break
        if not feasible:
            continue
        
        assignment = hungarian(cost)
        
        # Verify
        valid = True
        for i in range(n):
            if i == assignment[i] or cost[i][assignment[i]] == INF:
                valid = False
                break
        if not valid:
            continue
        
        cycles = find_cycles(assignment)
        
        if len(cycles) == 1:
            sc = cycle_score(cycles[0])
            if sc < best_score:
                best_score = sc
                best_cycle = cycles[0]
            # Try 2-opt improvement
            improved = two_opt_improve(best_cycle[:], deadline - 10.0)
            sc2 = cycle_score(improved)
            if sc2 < best_score:
                best_score = sc2
                best_cycle = improved
        else:
            # Repair: merge cycles
            merged = repair_cycle_cover(cycles, deadline - 10.0)
            if merged is not None:
                sc = cycle_score(merged)
                if sc < best_score:
                    best_score = sc
                    best_cycle = merged
                # Try 2-opt
                improved = two_opt_improve(merged[:], deadline - 10.0)
                sc2 = cycle_score(improved)
                if sc2 < best_score:
                    best_score = sc2
                    best_cycle = improved
    
    # Final 2-opt polish if we have time
    if best_cycle is not None and time.time() < deadline - 5.0:
        polished = two_opt_improve(best_cycle[:], deadline - 3.0)
        sc = cycle_score(polished)
        if sc < best_score:
            best_score = sc
            best_cycle = polished
    
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END

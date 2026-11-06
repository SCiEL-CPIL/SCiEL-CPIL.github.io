import time
import math
import random
import sys
import heapq
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
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

def _build_component_structure(triples, N):
    """
    Build the 2-section graph (edges between any two elements in the same triple)
    and find connected components. Each connected component is an independent
    conflict sub-problem: a solution for one component does not interact with
    another.
    """
    # 2-section graph: elements that co-occur in at least one triple
    adj = defaultdict(set)
    for (a, b, c) in triples:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)

    # Connected components via BFS
    visited = set()
    components = []
    for start in range(1, N + 1):
        if start in visited:
            continue
        comp = set()
        queue = [start]
        visited.add(start)
        while queue:
            u = queue.pop()
            comp.add(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        components.append(comp)

    # Sort components by size (smallest first for exhaustive solvability)
    components.sort(key=len)
    return components, adj

def _max_independent_set_exhaustive(elements, triples, time_limit=1.0):
    """
    Branch-and-bound maximum independent set for a small component.
    elements: list of elements in this component
    triples: list of triples restricted to this component (each triple is a frozenset of 3 elements)
    Returns the maximum independent set (as a set) and its size.
    """
    if not elements:
        return set(), 0

    # Build local index
    elem_list = sorted(elements)
    idx = {e: i for i, e in enumerate(elem_list)}
    n = len(elem_list)

    # For each element, list of triple indices it belongs to
    elem_triples = [[] for _ in range(n)]
    for t_idx, t in enumerate(triples):
        for e in t:
            elem_triples[idx[e]].append(t_idx)

    # Triple counts
    T = len(triples)
    counts = [0] * T

    # Sort elements by degree (descending) for better pruning
    order = sorted(range(n), key=lambda i: len(elem_triples[i]), reverse=True)

    best_set = set()
    best_size = 0
    current = [False] * n  # True if included
    start_time = time.time()

    # Precompute for each element, the set of triple indices (as a set for fast intersection)
    elem_triple_sets = [set(elem_triples[i]) for i in range(n)]

    def dfs(pos, included_count):
        nonlocal best_set, best_size, counts
        if time.time() - start_time > time_limit:
            return
        # Upper bound: current included + remaining positions
        if included_count + (n - pos) <= best_size:
            return
        if pos == n:
            if included_count > best_size:
                best_size = included_count
                best_set = set(elem_list[i] for i in range(n) if current[i])
            return

        i = order[pos]

        # Option 1: include element i
        can_include = True
        affected = []
        for t_idx in elem_triples[i]:
            if counts[t_idx] >= 2:
                can_include = False
                break
            affected.append(t_idx)
        if can_include:
            for t_idx in affected:
                counts[t_idx] += 1
            current[i] = True
            dfs(pos + 1, included_count + 1)
            current[i] = False
            for t_idx in affected:
                counts[t_idx] -= 1

        # Option 2: exclude element i
        dfs(pos + 1, included_count)

    dfs(0, 0)
    return best_set, best_size

def _greedy_removal_for_component(elements, triples, elem_triples_local, degree_local, seed, strategy):
    """
    Greedy removal within a single component.
    elements: list of elements in this component
    triples: list of triples restricted to this component
    elem_triples_local: dict mapping element -> list of local triple indices
    degree_local: dict mapping element -> number of triples it's in
    """
    random.seed(seed)
    S = set(elements)
    T = len(triples)
    counts = [3] * T

    active_count = {x: 0 for x in elements}
    for t_idx in range(T):
        a, b, c = triples[t_idx]
        active_count[a] += 1
        active_count[b] += 1
        active_count[c] += 1

    # Build priority
    def score(x):
        ac = active_count[x]
        if strategy == 0:
            return ac * 1000 + degree_local[x]
        elif strategy == 1:
            return ac
        else:
            return ac * 1000 + (180 - x)

    # Use a list-based approach for small components
    while True:
        # Find element with max score among those with active_count > 0
        best_x = None
        best_sc = -1
        candidates = []
        for x in S:
            if active_count[x] > 0:
                sc = score(x)
                if sc > best_sc:
                    best_sc = sc
                    best_x = x
                    candidates = [x]
                elif sc == best_sc:
                    candidates.append(x)

        if best_x is None or best_sc <= 0:
            break

        if strategy == 1 and len(candidates) > 1:
            best_x = random.choice(candidates)

        S.discard(best_x)

        for t_idx in elem_triples_local[best_x]:
            if counts[t_idx] == 3:
                a, b, c = triples[t_idx]
                active_count[a] -= 1
                active_count[b] -= 1
                active_count[c] -= 1
                counts[t_idx] = 2
            elif counts[t_idx] == 2:
                counts[t_idx] = 1

    return S

def _local_search_add_component(S, elements, triples, elem_triples_local, degree_local, seed):
    """
    Try to add back removed elements within a component.
    """
    random.seed(seed)
    S = set(S)
    T = len(triples)
    counts = [0] * T
    for x in S:
        for t_idx in elem_triples_local[x]:
            counts[t_idx] += 1

    removed = [x for x in elements if x not in S]
    removed.sort(key=lambda x: degree_local[x])

    max_passes = 5
    for _ in range(max_passes):
        improved = False
        for x in removed:
            if x in S:
                continue
            can_add = True
            for t_idx in elem_triples_local[x]:
                if counts[t_idx] >= 2:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                improved = True
                for t_idx in elem_triples_local[x]:
                    counts[t_idx] += 1
        if not improved:
            break

    return S

def _two_opt_swap_component(S, elements, triples, elem_triples_local, degree_local, seed, max_attempts=30):
    """
    2-opt swap moves within a component.
    """
    random.seed(seed)
    S = set(S)
    T = len(triples)
    counts = [0] * T
    for x in S:
        for t_idx in elem_triples_local[x]:
            counts[t_idx] += 1

    removed = [x for x in elements if x not in S]
    if not removed or not S:
        return S

    removed.sort(key=lambda x: degree_local[x])
    best_S = set(S)
    best_size = len(S)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        s_list = list(S)
        weights = [degree_local[x] + 1 for x in s_list]
        total_weight = sum(weights)
        r = random.random() * total_weight
        cumulative = 0
        x_remove = s_list[0]
        for x, w in zip(s_list, weights):
            cumulative += w
            if r <= cumulative:
                x_remove = x
                break

        S.discard(x_remove)
        for t_idx in elem_triples_local[x_remove]:
            counts[t_idx] -= 1

        for y in removed:
            if y in S:
                continue
            can_add = True
            for t_idx in elem_triples_local[y]:
                if counts[t_idx] >= 2:
                    can_add = False
                    break
            if can_add:
                S.add(y)
                for t_idx in elem_triples_local[y]:
                    counts[t_idx] += 1

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            removed = [x for x in elements if x not in best_S]
            removed.sort(key=lambda x: degree_local[x])
        else:
            S = set(best_S)
            counts = [0] * T
            for x in best_S:
                for t_idx in elem_triples_local[x]:
                    counts[t_idx] += 1
            removed = [x for x in elements if x not in best_S]
            removed.sort(key=lambda x: degree_local[x])

    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 87.0
    N = 180

    triples = _precompute_triples(N)
    T = len(triples)

    # Build global element-to-triple index
    elem_triples_global = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        elem_triples_global[a].append(idx)
        elem_triples_global[b].append(idx)
        elem_triples_global[c].append(idx)

    # Build component structure
    components, adj = _build_component_structure(triples, N)

    # For each component, prepare local data
    component_data = []
    for comp in components:
        comp_set = set(comp)
        # Local triples
        local_triples = []
        local_elem_triples = defaultdict(list)
        for idx, (a, b, c) in enumerate(triples):
            if a in comp_set and b in comp_set and c in comp_set:
                local_triples.append((a, b, c))
                local_elem_triples[a].append(len(local_triples) - 1)
                local_elem_triples[b].append(len(local_triples) - 1)
                local_elem_triples[c].append(len(local_triples) - 1)
        degree_local = {x: len(local_elem_triples[x]) for x in comp}
        component_data.append({
            'elements': sorted(comp),
            'triples': local_triples,
            'elem_triples': local_elem_triples,
            'degree': degree_local,
            'size': len(comp)
        })

    best_set = set()
    best_size = 0

    # Phase 1: Solve each component
    for ci, cdata in enumerate(component_data):
        if time.time() > deadline - 5.0:
            break

        elements = cdata['elements']
        local_triples = cdata['triples']
        local_elem_triples = cdata['elem_triples']
        degree_local = cdata['degree']
        comp_size = cdata['size']

        comp_best = set()
        comp_best_size = 0

        # For small components, try exhaustive search
        if comp_size <= 25 and len(local_triples) <= 80:
            t_limit = min(2.0, max(0.1, (deadline - time.time() - 5.0) / max(1, len([c for c in component_data if c['size'] <= 25]))))
            es, es_size = _max_independent_set_exhaustive(elements, local_triples, t_limit)
            if es_size > comp_best_size:
                comp_best = es
                comp_best_size = es_size

        # Multiple greedy restarts for this component
        num_restarts = min(50, max(5, int((deadline - time.time() - 5.0) / 0.05)))
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            strategy = r % 3
            seed = ci * 1000 + r * 137

            S_greedy = _greedy_removal_for_component(elements, local_triples, local_elem_triples, degree_local, seed, strategy)
            S_added = _local_search_add_component(S_greedy, elements, local_triples, local_elem_triples, degree_local, seed + 100)
            S_final = _two_opt_swap_component(S_added, elements, local_triples, local_elem_triples, degree_local, seed + 200, max_attempts=20)

            if len(S_final) > comp_best_size:
                comp_best_size = len(S_final)
                comp_best = set(S_final)

        best_set.update(comp_best)
        best_size += comp_best_size

    # Phase 2: Global polish - try to add any elements not yet in best_set
    if time.time() < deadline - 3.0:
        counts = [0] * T
        for x in best_set:
            for t_idx in elem_triples_global[x]:
                counts[t_idx] += 1

        remaining = [x for x in range(1, N + 1) if x not in best_set]
        remaining.sort(key=lambda x: len(elem_triples_global[x]))

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in remaining:
                if x in best_set:
                    continue
                can_add = True
                for t_idx in elem_triples_global[x]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    best_set.add(x)
                    improved = True
                    for t_idx in elem_triples_global[x]:
                        counts[t_idx] += 1

    # Final verification: remove any violating triples
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            best_set.discard(c)
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            best_set.discard(a)
            best_set.discard(b)
            best_set.discard(c)

    if not best_set:
        best_set = set(range(90, 181))

    return list(best_set)

# EVOLVE-BLOCK-END

"""Longest simple path in the pinned G(100, 1.8/100) graph (seed 748).
Strategy:
  1. Build components. For each component, try to find a Hamiltonian path.
  2. For tree components, use leaf-peeling.
  3. For general components, use a combination of:
     - Maximum matching to get a strong seed path
     - Iterative deepening DFS with Warnsdorff ordering and strong pruning
     - Multiple restarts from different start vertices
  4. Return the best path found before the deadline.
"""
import time
import random
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18],
         [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13],
         [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18],
         [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76],
         [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52],
         [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55],
         [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70],
         [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42],
         [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48],
         [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78],
         [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80],
         [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81],
         [69, 90], [70, 92], [74, 85], [75, 82]]

# Build adjacency
ADJ_SET = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ_SET[u].add(v)
    ADJ_SET[v].add(u)
ADJ_SORTED = [sorted(s) for s in ADJ_SET]
DEG = [len(s) for s in ADJ_SET]


def components():
    """Return list of components (each a set), sorted by size desc."""
    seen = set()
    comps = []
    for s in range(N):
        if s in seen:
            continue
        comp = set()
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in ADJ_SET[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def is_tree(comp):
    """A connected component is a tree iff edges == |comp|-1."""
    edges = 0
    for u in comp:
        edges += len(ADJ_SET[u])
    edges //= 2
    return edges == len(comp) - 1


def leaf_order_path(comp):
    """For a tree component, build a Hamiltonian path via leaf peeling."""
    if not comp:
        return None
    remaining = set(comp)
    deg = {u: len(ADJ_SET[u] & remaining) for u in remaining}
    leaves = [u for u in remaining if deg[u] <= 1]
    if not leaves:
        return None
    start = min(leaves)
    path = [start]
    pathset = {start}
    while len(path) < len(comp):
        cur = path[-1]
        cand = [v for v in ADJ_SET[cur] if v not in pathset]
        if not cand:
            return None
        # Choose candidate that keeps unvisited connected
        unvisited = comp - pathset
        chosen = None
        for v in cand:
            rest = unvisited - {v}
            if not rest:
                chosen = v
                break
            seed = None
            for w in ADJ_SET[v]:
                if w in rest:
                    seed = w
                    break
            if seed is None:
                continue
            seenr = {seed}
            st = [seed]
            while st:
                x = st.pop()
                for y in ADJ_SET[x]:
                    if y in rest and y not in seenr:
                        seenr.add(y)
                        st.append(y)
            if len(seenr) == len(rest):
                chosen = v
                break
        if chosen is None:
            if cand:
                chosen = cand[0]
            else:
                return None
        path.append(chosen)
        pathset.add(chosen)
    for i in range(len(path) - 1):
        if path[i + 1] not in ADJ_SET[path[i]]:
            return None
    return path


def warnsdorff_extend(path, pathset, deadline):
    """Greedy Warnsdorff extension from both ends."""
    path = list(path)
    pathset = set(path)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # extend front
        while True:
            if time.time() > deadline:
                break
            cur = path[0]
            cand = [v for v in ADJ_SET[cur] if v not in pathset]
            if not cand:
                break
            def score(w):
                return sum(1 for z in ADJ_SET[w] if z not in pathset)
            v = min(cand, key=lambda w: (score(w), w))
            path.insert(0, v)
            pathset.add(v)
            improved = True
        # extend back
        while True:
            if time.time() > deadline:
                break
            cur = path[-1]
            cand = [v for v in ADJ_SET[cur] if v not in pathset]
            if not cand:
                break
            def score(w):
                return sum(1 for z in ADJ_SET[w] if z not in pathset)
            v = min(cand, key=lambda w: (score(w), w))
            path.append(v)
            pathset.add(v)
            improved = True
    return path


def matching_seed_path(comp, deadline):
    """Use maximum matching to construct a seed path."""
    comp_list = sorted(comp)
    if len(comp_list) <= 2:
        return list(comp_list)
    
    # Try to build a path using a greedy matching approach
    # Start from a leaf (degree 1 in the component)
    comp_set = set(comp)
    leaves = [u for u in comp_list if len(ADJ_SET[u] & comp_set) == 1]
    if not leaves:
        leaves = [min(comp_list)]
    
    start = leaves[0]
    path = [start]
    pathset = {start}
    path = warnsdorff_extend(path, pathset, deadline)
    return path


def dfs_longest_path(start, comp, comp_size, deadline, best):
    """DFS from start vertex, Warnsdorff-ordered, with pruning."""
    best = list(best)
    
    def dfs(path, pathset):
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best.clear()
            best.extend(path)
        if len(path) >= comp_size:
            return
        cur = path[-1]
        cand = [v for v in ADJ_SORTED[cur] if v not in pathset]
        if not cand:
            return
        # Warnsdorff ordering
        def onward(w):
            return sum(1 for z in ADJ_SET[w] if z not in pathset)
        cand.sort(key=lambda w: (onward(w), w))
        for v in cand:
            if time.time() > deadline:
                return
            path.append(v)
            pathset.add(v)
            dfs(path, pathset)
            path.pop()
            pathset.remove(v)
    
    dfs([start], {start})
    return best


def multi_start_search(comp, comp_size, deadline, best):
    """Try many starts with Warnsdorff extend + bounded DFS, keep best."""
    best = list(best)
    comp_list = sorted(comp)
    comp_set = set(comp)
    
    # Order starts by degree (leaves first)
    comp_list.sort(key=lambda u: (len(ADJ_SET[u] & comp_set), u))
    
    for start in comp_list:
        if time.time() > deadline:
            break
        # Quick Warnsdorff extend
        seed_path = [start]
        seed_path = warnsdorff_extend(seed_path, {start}, deadline)
        if len(seed_path) > len(best):
            best = list(seed_path)
        if len(seed_path) >= comp_size:
            break
        
        # Bounded DFS
        per = min(0.5, max(0.05, (deadline - time.time()) * 0.15))
        dl = min(deadline, time.time() + per)
        best = dfs_longest_path(start, comp, comp_size, dl, best)
        if len(best) >= comp_size:
            break
    
    return best


def verify(path):
    if len(path) < 1:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in ADJ_SET[path[i]]:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # Safety margin before 90s hard limit

    comps = components()
    best = []

    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= len(best):
            continue

        # Fast exact case: tree component
        if is_tree(comp):
            p = leaf_order_path(comp)
            if p is not None and len(p) == comp_size and verify(p):
                if len(p) > len(best):
                    best = p
                continue

        # General component: multi-start search
        best = multi_start_search(comp, comp_size, deadline, best)
        if len(best) >= comp_size:
            break

    if not best:
        best = [0]
    if not verify(best):
        for v in range(N):
            best = [v]
            break
    return best

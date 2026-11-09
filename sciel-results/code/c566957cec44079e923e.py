"""Longest simple path in pinned G(100, 1.8/100) graph via skeleton decomposition + DFS + local improvement."""
import time
import random
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def build_skeleton(comp):
    """
    Decompose component into skeleton:
    - Branching nodes (degree >= 3)
    - Chains of degree-2 nodes between branching nodes
    Returns:
      - branch_nodes: list of branching node IDs
      - chains: dict mapping (a,b) -> list of internal vertices (excluding endpoints)
      - chain_edge_count: dict mapping (a,b) -> number of edges in the chain
    """
    comp_set = set(comp)
    branch_nodes = [v for v in comp if DEG[v] >= 3]
    
    if not branch_nodes:
        return comp, {}, {}
    
    chains = {}
    chain_edge_count = {}
    processed_edges = set()
    
    for bn in branch_nodes:
        for nb in sorted(ADJ[bn]):
            if nb not in comp_set:
                continue
            edge_key = tuple(sorted([bn, nb]))
            if edge_key in processed_edges:
                continue
            
            # Walk the chain from nb
            chain_verts = []
            prev = bn
            curr = nb
            edge_count = 1
            
            while curr not in branch_nodes or curr == bn:
                if curr in branch_nodes and curr != bn:
                    break
                chain_verts.append(curr)
                # Find next
                nexts = [x for x in ADJ[curr] if x != prev]
                if not nexts:
                    break
                prev = curr
                curr = nexts[0]
                edge_count += 1
            
            if curr in branch_nodes and curr != bn:
                key = tuple(sorted([bn, curr]))
                if key not in chains:
                    chains[key] = chain_verts
                    chain_edge_count[key] = edge_count
                    for v in chain_verts:
                        for e in ADJ[v]:
                            ek = tuple(sorted([v, e]))
                            processed_edges.add(ek)
            elif curr in branch_nodes and curr == bn:
                # Cycle back to same branch node
                key = (bn, bn)
                if key not in chains:
                    chains[key] = chain_verts
                    chain_edge_count[key] = edge_count
    
    return branch_nodes, chains, chain_edge_count

def skeleton_longest_path(branch_nodes, chains, chain_edge_count, deadline):
    """
    Find longest path on the skeleton using DFS.
    Each chain traversal adds the internal vertices.
    We track which chains have been used.
    """
    if not branch_nodes:
        return []
    
    # Build skeleton adjacency
    skel_adj = {bn: [] for bn in branch_nodes}
    for key, verts in chains.items():
        a, b = key
        if a != b:
            skel_adj[a].append((b, key))
            skel_adj[b].append((a, key))
        else:
            skel_adj[a].append((a, key))
    
    best_path = []
    
    def dfs(current, path_verts, used_chains, visited_nodes):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path_verts) > len(best_path):
            best_path = path_verts[:]
        
        for nb, key in skel_adj[current]:
            if key in used_chains:
                continue
            if nb == current:
                # Self-loop chain
                internal = chains[key]
                if all(v not in visited_nodes for v in internal):
                    new_path = path_verts + internal
                    dfs(current, new_path, used_chains | {key}, visited_nodes | set(internal))
            else:
                if nb in visited_nodes:
                    continue
                internal = chains[key]
                if all(v not in visited_nodes for v in internal):
                    new_path = path_verts + internal + [nb]
                    dfs(nb, new_path, used_chains | {key}, visited_nodes | {nb} | set(internal))
    
    for start in branch_nodes:
        if time.time() > deadline:
            break
        dfs(start, [start], set(), {start})
    
    return best_path

def greedy_warnsdorff(start, seen):
    """Warnsdorff's rule: pick neighbor with fewest unvisited neighbors."""
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def greedy_max_degree(start, seen):
    """Pick neighbor with highest degree."""
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (DEG[w], w))
        path.append(v)
        seen.add(v)
    return path

def greedy_random(start, seen, rng):
    """Random neighbor choice."""
    path = [start]
    while True:
        cand = list(ADJ[path[-1]] - seen)
        if not cand:
            break
        v = rng.choice(cand)
        path.append(v)
        seen.add(v)
    return path

def greedy_mixed(start, seen, rng):
    """Mixed heuristic."""
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        w = rng.random()
        v = max(cand, key=lambda x: (-len(ADJ[x] - seen), DEG[x] * w, x))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def local_improve(path, seen, deadline):
    """Try to insert unvisited vertices into the path and extend ends."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in list(unvisited):
            found = False
            best_pos = -1
            best_score = float('inf')
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    score = len(ADJ[u] - seen - {path[i], path[i+1]})
                    if score < best_score:
                        best_pos = i + 1
                        best_score = score
            if best_pos >= 0:
                path.insert(best_pos, u)
                seen.add(u)
                unvisited.remove(u)
                improved = True
                found = True
            if not found:
                if path[0] in ADJ[u]:
                    path.insert(0, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                elif path[-1] in ADJ[u]:
                    path.append(u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
        # Extend ends
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def dfs_path_search(start, seen, path, best, deadline, node_count, max_nodes=500000):
    """DFS to find longer path with Warnsdorff ordering."""
    if time.time() > deadline or node_count[0] > max_nodes:
        return
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    cand = ADJ[path[-1]] - seen
    cand = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    for v in cand:
        node_count[0] += 1
        path.append(v)
        seen.add(v)
        dfs_path_search(start, seen, path, best, deadline, node_count, max_nodes)
        path.pop()
        seen.remove(v)
        if time.time() > deadline or node_count[0] > max_nodes:
            return

def construct():
    t0 = time.time()
    deadline = t0 + 80  # Safety margin before 90s kill
    
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    rng = random.Random(42)
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        
        # Phase 1: Skeleton-based search
        branch_nodes, chains, chain_edge_count = build_skeleton(comp)
        if branch_nodes:
            skel_path = skeleton_longest_path(branch_nodes, chains, chain_edge_count, deadline)
            if len(skel_path) > len(best_path):
                best_path = skel_path[:]
        
        # Phase 2: Diverse greedy starts from every vertex
        for start in comp_list:
            if time.time() > deadline:
                break
            for func in [greedy_warnsdorff, greedy_max_degree, greedy_mixed]:
                if time.time() > deadline:
                    break
                seen = {start}
                p = func(start, seen, rng) if func != greedy_warnsdorff and func != greedy_max_degree else func(start, seen)
                p = extend_both_ends(p, set(p))
                if len(p) > len(best_path):
                    best_path = p[:]
            # Also random
            seen = {start}
            p = greedy_random(start, seen, rng)
            p = extend_both_ends(p, set(p))
            if len(p) > len(best_path):
                best_path = p[:]
        
        # Phase 3: Local improvement
        if best_path and len(best_path) < comp_size:
            if time.time() < deadline - 2:
                p = local_improve(best_path, set(best_path), deadline - 2)
                if len(p) > len(best_path):
                    best_path = p
        
        # Phase 4: DFS from promising starts if not Hamiltonian
        if len(best_path) < comp_size:
            remaining = deadline - time.time()
            if remaining > 1:
                # Try from endpoints of best path and a few others
                starts_to_try = list(set([best_path[0], best_path[-1]] + comp_list[:10]))
                for start in starts_to_try:
                    if time.time() > deadline - 1:
                        break
                    rem = deadline - time.time()
                    if rem < 0.5:
                        break
                    budget = min(3.0, rem * 0.4)
                    dl = time.time() + budget
                    node_count = [0]
                    local_best = list(best_path)
                    dfs_path_search(start, {start}, [start], local_best, dl, node_count, max_nodes=200000)
                    if len(local_best) > len(best_path):
                        best_path = local_best[:]
                        if len(best_path) < comp_size and time.time() < deadline - 1:
                            p2 = local_improve(best_path, set(best_path), deadline - 1)
                            if len(p2) > len(best_path):
                                best_path = p2
                        if len(best_path) >= comp_size:
                            break
    
    # Final polish
    if best_path and len(best_path) < max(len(c) for c in comps) and time.time() < deadline - 1:
        p_final = local_improve(best_path, set(best_path), deadline - 0.5)
        if len(p_final) > len(best_path):
            best_path = p_final
    
    # Verify
    if not best_path:
        best_path = [0]
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path

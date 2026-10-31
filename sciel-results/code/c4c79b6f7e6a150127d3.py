import time
import math
import random
import numpy as np
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    def build_collision_graph(U):
        """Build a graph where edges represent sum collisions.
        For each pair (a,b) with a+b = s, if another pair (c,d) also has c+d=s,
        we add an edge between the pairs. We model this as a vertex cover problem
        where vertices are elements and edges connect elements that participate
        in the same sum collision."""
        n = len(U)
        # Map each sum to the list of (i,j) pairs that produce it
        sum_to_pairs = defaultdict(list)
        for i in range(n):
            for j in range(i, n):  # symmetric, a+b = b+a
                s = U[i] + U[j]
                sum_to_pairs[s].append((i, j))
        
        # Build adjacency: element i is connected to element j if they share a sum collision
        # i.e., there exists s such that i appears in some pair for s and j appears in another pair for s
        adj = defaultdict(set)
        for s, pairs in sum_to_pairs.items():
            if len(pairs) > 1:
                # All elements involved in this sum are connected
                elements = set()
                for i, j in pairs:
                    elements.add(i)
                    elements.add(j)
                el_list = list(elements)
                for i in range(len(el_list)):
                    for j in range(i + 1, len(el_list)):
                        adj[el_list[i]].add(el_list[j])
                        adj[el_list[j]].add(el_list[i])
        
        return adj
    
    def min_vertex_cover_heuristic(U, adj, max_iter=50):
        """Greedy minimum vertex cover: repeatedly remove the vertex with highest degree
        that is part of the most collisions, until no collisions remain or max_iter reached."""
        n = len(U)
        remaining = set(range(n))
        removed = []
        
        for _ in range(max_iter):
            if not remaining:
                break
            
            # Find the vertex in remaining with highest "collision degree"
            # (number of edges to other vertices in remaining)
            best_v = -1
            best_deg = 0
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1 or best_deg == 0:
                break
            
            removed.append(best_v)
            remaining.discard(best_v)
        
        # Return the set of elements NOT removed
        result = [U[i] for i in remaining]
        if 0 not in result:
            result.append(0)
        return sorted(set(result))
    
    def max_flow_min_cut_for_collisions(U, adj):
        """Use max-flow to find minimum vertex cover in the collision graph.
        Transform bipartite graph: split each vertex v into v_in and v_out.
        Edge v_in -> v_out has capacity 1.
        For each edge (u,v) in original, add edge u_out -> v_in and v_out -> u_in with capacity INF.
        Min cut gives min vertex cover."""
        n = len(U)
        # Nodes: source, sink, v_in (0..n-1), v_out (n..2n-1)
        src = 2 * n
        sink = 2 * n + 1
        total_nodes = 2 * n + 2
        
        INF = 10**9
        
        # Build residual graph
        # adj_residual[i][j] = capacity
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        
        # Source to v_in for each vertex
        for v in range(n):
            cap[src][v] = 1
        
        # v_out to sink for each vertex
        for v in range(n):
            cap[n + v][sink] = 1
        
        # For each edge (u,v) in collision graph: u_out -> v_in and v_out -> u_in
        for u in range(n):
            for v in adj[u]:
                if u < v:  # avoid duplicates
                    cap[n + u][v] = INF
                    cap[n + v][u] = INF
        
        # Ford-Fulkerson with BFS (Edmonds-Karp)
        def bfs():
            parent = [-1] * total_nodes
            visited = [False] * total_nodes
            q = deque([src])
            visited[src] = True
            while q:
                u = q.popleft()
                for v in range(total_nodes):
                    if not visited[v] and cap[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return parent
                        q.append(v)
            return None
        
        total_flow = 0
        while True:
            parent = bfs()
            if parent is None:
                break
            # Find min capacity along path
            path_cap = INF
            v = sink
            while v != src:
                u = parent[v]
                path_cap = min(path_cap, cap[u][v])
                v = u
            # Update residual
            v = sink
            while v != src:
                u = parent[v]
                cap[u][v] -= path_cap
                cap[v][u] += path_cap
                v = u
            total_flow += path_cap
        
        # Min vertex cover: vertices v where v_in is on source side and v_out is on sink side
        # Actually: v is in cover if v_in reachable from source in residual AND v_out not reachable
        # Simpler: v is in cover if there's a path from source to v_in but not from v_out to sink
        # Let's just do: v in cover if v_in is reachable from src in final residual
        visited = [False] * total_nodes
        q = deque([src])
        visited[src] = True
        while q:
            u = q.popleft()
            for v in range(total_nodes):
                if not visited[v] and cap[u][v] > 0:
                    visited[v] = True
                    q.append(v)
        
        cover = set()
        for v in range(n):
            # v_in is node v, v_out is node n+v
            if visited[v] and not visited[n + v]:
                cover.add(v)
        
        result = [U[i] for i in range(n) if i not in cover]
        if 0 not in result:
            result.append(0)
        return sorted(set(result))
    
    # Phase 1: Try known good Golomb rulers as initial candidates
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 31, 34],
        [0, 1, 4, 10, 18, 23, 31, 42, 45],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 54],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 65],
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline - 15:
            break
        try_set(gr)
    
    # Phase 2: Exhaustive search for small M
    for M in range(1, 13):
        if time.time() > deadline - 15:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 15:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 3: Max-flow based collision elimination
    # Start from a moderately large candidate set and use min-cut to reduce collisions
    if time.time() < deadline - 20:
        # Generate diverse starting sets
        starting_sets = []
        
        # Arithmetic progressions with various steps
        for step in [1, 2, 3, 5, 7, 11]:
            for max_val in [20, 40, 60, 100, 150]:
                ap = list(range(0, max_val + 1, step))
                if 0 in ap and len(ap) >= 3:
                    starting_sets.append(ap)
        
        # Geometric-like sets
        for ratio in [2, 3, 4]:
            g = [0, 1]
            while g[-1] * ratio <= 200:
                g.append(g[-1] * ratio)
            starting_sets.append(g)
        
        # Random structured sets
        random.seed(42)
        for _ in range(30):
            max_val = random.randint(30, 150)
            n = random.randint(5, 12)
            inner = random.sample(range(1, max_val), min(n - 1, max_val - 1))
            cand = sorted([0] + inner + [max_val])
            starting_sets.append(cand)
        
        # For each starting set, try min-cut collision elimination
        for start_set in starting_sets:
            if time.time() > deadline - 15:
                break
            
            try_set(start_set)
            
            # Build collision graph and try min-cut
            if len(start_set) >= 4:
                adj = build_collision_graph(start_set)
                
                # Try greedy vertex cover removal
                for max_iter in [3, 5, 8, 12]:
                    if time.time() > deadline - 15:
                        break
                    reduced = min_vertex_cover_heuristic(start_set, adj, max_iter)
                    if len(reduced) >= 2:
                        try_set(reduced)
                
                # Try max-flow min-cut (more expensive but potentially better)
                if len(start_set) <= 25 and time.time() < deadline - 15:
                    reduced = max_flow_min_cut_for_collisions(start_set, adj)
                    if len(reduced) >= 2:
                        try_set(reduced)
    
    # Phase 4: Heuristic local search with multiple strategies
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current) if current else 0
        
        # Strategy 1: Try adding an element
        if not improved:
            candidates_to_add = set()
            for i in range(1, max_cur + 40):
                if i not in current:
                    candidates_to_add.add(i)
            if len(current) > 2:
                for _ in range(40):
                    candidates_to_add.add(random.randint(max_cur + 1, max_cur + 250))
            for _ in range(20):
                candidates_to_add.add(random.randint(1, max_cur + 80))
                
            for new_el in sorted(candidates_to_add):
                if time.time() > deadline - 10:
                    break
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 2: Try replacing an element
        if not improved and len(current) >= 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                old_el = current[i]
                for delta in range(-40, 41):
                    if time.time() > deadline - 10:
                        break
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Strategy 3: Try removing an element
        if not improved and len(current) > 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 4: Collision-guided removal
        # If we have collisions, try removing the element involved in most collisions
        if not improved and len(current) >= 5:
            adj = build_collision_graph(current)
            # Find element with highest collision degree
            max_deg = 0
            worst_idx = -1
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                deg = len(adj[i])
                if deg > max_deg:
                    max_deg = deg
                    worst_idx = i
            
            if worst_idx >= 0 and max_deg > 0:
                cand = [x for j, x in enumerate(current) if j != worst_idx]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
        
        # Strategy 5: Try swapping two elements
        if not improved and len(current) > 3:
            for i in range(1, len(current)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 10:
                        break
                    cand = list(current)
                    cand[i], cand[j] = cand[j], cand[i]
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local optima
            if random.random() < 0.6:
                perturbation_type = random.random()
                if perturbation_type < 0.25:
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        j = random.randint(1, len(current)-1)
                        if i != j:
                            cand = list(current)
                            cand[i], cand[j] = cand[j], cand[i]
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                elif perturbation_type < 0.5:
                    new_el = random.randint(1, max_cur + 120)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
                elif perturbation_type < 0.75:
                    if len(current) > 1:
                        i = random.randint(1, len(current)-1)
                        new_el = random.randint(1, max_cur + 60)
                        if new_el not in current:
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                else:
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        cand = [x for j, x in enumerate(current) if j != i]
                        try_set(cand)
                        current = list(best_set)
    
    return list(best_set)

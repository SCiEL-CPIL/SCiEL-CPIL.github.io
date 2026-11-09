import time
import random
import numpy as np
from collections import defaultdict
from scipy.optimize import linprog

def construct():
    t0 = time.time()
    DEADLINE = 148.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_mcf(edges, deadline):
        """
        Decompose graph into minimum number of edge-disjoint paths using MCF.
        """
        n = max([v for e in edges for v in e] + [0]) + 1
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Degrees for lower bound and sanity
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[b] += 1 # Typo in thought, fix below
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        lb = sum(1 for d in deg.values() if d % 2 == 1) // 2
        if lb == 0 and edges:
            lb = 1 # If Eulerian, 1 path (cycle) or 0 if no edges. But we need paths covering edges.
                   # Actually if all even, we can have 1 cycle. But a cycle is not a path in strict sense?
                   # Usually "path" in these contexts allows cycles or we break a cycle.
                   # The prompt says "simple paths". A cycle is not a simple path (start != end).
                   # So we must break at least one edge if there's a cycle.
                   # So lb is at least 1 if edges exist.

        # MCF Formulation:
        # Variables x_{uv} in {0,1} for each directed arc (u,v) corresponding to undirected edge {u,v}.
        # We create arcs (u,v) and (v,u) for each {u,v}.
        # Minimize sum x_{uv} subject to:
        # 1. Flow conservation: For each node i, sum_in x - sum_out x = b_i
        #    where b_i is the "imbalance".
        #    For a set of paths, the nodes with b_i != 0 are the endpoints.
        #    Sum(b_i) must be 0.
        #    We want to minimize the number of paths.
        #    Number of paths = (1/2) * sum_{i: b_i != 0} |b_i|? 
        #    Actually, if we allow paths to have length > 0, each path has 2 endpoints.
        #    So total paths = (1/2) * sum |b_i| if we restrict b_i to {-1, 0, 1}?
        #    Wait, if a node is an endpoint of multiple paths, b_i could be > 1.
        #    To minimize number of paths, we should prefer b_i in {-1, 0, 1} if possible?
        #    No, minimizing sum |b_i| / 2 is equivalent to minimizing number of endpoints, 
        #    which is 2 * number of paths.
        #    
        #    Constraints:
        #    For each edge {u,v}, x_{uv} + x_{vu} = 1 (must use the edge in one direction).
        #    Flow conservation: sum_{v: (u,v) in arcs} x_{uv} - sum_{v: (v,u) in arcs} x_{vu} = b_u
        #    
        #    We need to choose b_u.
        #    b_u must satisfy:
        #    1. sum b_u = 0
        #    2. b_u <= deg(u)
        #    3. b_u >= -deg(u)
        #    4. b_u has same parity as deg(u)? 
        #       In a flow decomposition into paths, the degree of a node in the subgraph formed by 
        #       selected arcs is deg(u). 
        #       Let d_in(u) + d_out(u) = deg(u).
        #       b_u = d_out(u) - d_in(u).
        #       So b_u + d_in(u) = d_out(u) and d_in(u) + d_out(u) = deg(u).
        #       2*d_in(u) = deg(u) - b_u.
        #       So deg(u) - b_u must be even. Thus b_u % 2 == deg(u) % 2.
        #    
        #    Objective: Minimize 0.5 * sum |b_u|
        #    This is a linear objective if we split b_u = b_u^+ - b_u^- with b_u^+, b_u^- >= 0.
        #    Minimize sum (b_u^+ + b_u^-).
        
        # Build LP
        # Variables:
        # 1. x_{uv} for each directed arc. (2 * |E| variables)
        # 2. b_u^+, b_u^- for each node. (2 * n variables)
        
        arcs = []
        arc_idx = {}
        for u, v in edges:
            arcs.append((u, v))
            arcs.append((v, u))
            arc_idx[(u, v)] = len(arcs) - 1
            arc_idx[(v, u)] = len(arcs) - 1
            
        m_arcs = len(arcs)
        m_nodes = n
        total_vars = m_arcs + 2 * m_nodes
        
        # Objective: Minimize sum(b_u^+ + b_u^-)
        c = np.zeros(total_vars)
        for i in range(m_nodes):
            c[m_arcs + 2*i] = 1.0      # b_i^+
            c[m_arcs + 2*i + 1] = 1.0  # b_i^-
            
        # Constraints:
        # 1. For each edge {u,v}: x_{uv} + x_{vu} = 1
        A_eq = []
        b_eq = []
        
        for u, v in edges:
            row = np.zeros(total_vars)
            row[arc_idx[(u, v)]] = 1.0
            row[arc_idx[(v, u)]] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
            
        # 2. Flow conservation: sum_out x - sum_in x - b_u^+ + b_u^- = 0
        #    => sum_out x - sum_in x = b_u^+ - b_u^-
        #    => sum_out x - sum_in x - b_u^+ + b_u^- = 0
        for u in range(n):
            row = np.zeros(total_vars)
            # sum_out
            for v in adj[u]:
                if (u, v) in arc_idx:
                    row[arc_idx[(u, v)]] = 1.0
            # sum_in
            for v in adj[u]:
                if (v, u) in arc_idx:
                    row[arc_idx[(v, u)]] = -1.0
            # - b_u^+ + b_u^-
            row[m_arcs + 2*u] = -1.0
            row[m_arcs + 2*u + 1] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)
            
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds:
        # x_{uv} in [0, 1]
        # b_u^+, b_u^- in [0, deg(u)]
        bounds = [(0, 1)] * m_arcs
        for u in range(n):
            bounds.append((0, deg[u]))
            bounds.append((0, deg[u]))
            
        # Solve
        # Use Highs if available, else CBC or others.
        # linprog default is 'highs' in newer scipy.
        try:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        except Exception:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
            
        if not res.success:
            # Fallback to heuristic if LP fails
            return None
            
        # Extract solution
        # x values are integer (0 or 1) ideally.
        x_vals = res.x[:m_arcs]
        b_plus = res.x[m_arcs : m_arcs + m_nodes]
        b_minus = res.x[m_arcs + m_nodes : m_arcs + 2*m_nodes]
        
        # Reconstruct directed edges
        directed_edges = []
        for (u, v), idx in arc_idx.items():
            if x_vals[idx] > 0.5:
                directed_edges.append((u, v))
                
        # Verify coverage and disjointness
        covered = set()
        for u, v in directed_edges:
            e = (min(u, v), max(u, v))
            if e in covered:
                return None # Conflict
            covered.add(e)
        if covered != set(edges):
            return None # Missing edges
            
        # Reconstruct paths from directed edges
        # The directed graph is a collection of paths and cycles.
        # We need to break cycles into paths.
        # Start from nodes with b > 0 (sources) or b < 0 (sinks)?
        # b_u = out - in.
        # Source: b > 0. Sink: b < 0.
        # In a path u->v->w, source u has b=1, sink w has b=-1, intermediate v has b=0.
        
        out_adj = defaultdict(list)
        in_adj = defaultdict(list)
        for u, v in directed_edges:
            out_adj[u].append(v)
            in_adj[v].append(u)
            
        # Find starting points
        # Nodes with out_degree > in_degree are starts.
        # Nodes with out_degree < in_degree are ends.
        # Nodes with out_degree == in_degree are internal or part of cycles.
        
        starts = []
        for u in range(n):
            out_deg = len(out_adj[u])
            in_deg = len(in_adj[u])
            if out_deg > in_deg:
                # This node is a start of (out_deg - in_deg) paths
                starts.extend([u] * (out_deg - in_deg))
                
        # If no starts, we have cycles. We must break them.
        # Pick any node in a cycle and start there, breaking the cycle.
        if not starts:
            # Find a cycle
            visited = set()
            for u in range(n):
                if out_adj[u] and u not in visited:
                    # Follow path
                    curr = u
                    path = [curr]
                    visited.add(curr)
                    while out_adj[curr]:
                        # Pick one
                        nxt = out_adj[curr][0]
                        curr = nxt
                        path.append(curr)
                        visited.add(curr)
                        if len(path) > 1000: break
                    if path[0] == path[-1]: # Cycle found
                        # Break it: start at path[0], end at path[-2]
                        # This creates one path.
                        starts.append(path[0])
                        break
                    elif len(path) > 1:
                        # Path found? But we said no starts. This implies we entered a cycle from a path?
                        # No, if out==in for all, it's all cycles.
                        # If we found a path, it must have ended at a node with no out?
                        # But out==in means if in>0 then out>0.
                        # So we must have looped.
                        pass
            
        paths = []
        used_edges = set()
        
        for start in starts:
            # Follow path
            curr = start
            path = [curr]
            while out_adj[curr]:
                # Find an unused outgoing edge
                next_node = None
                for v in out_adj[curr]:
                    e = (curr, v)
                    if e not in used_edges:
                        next_node = v
                        used_edges.add(e)
                        break
                if next_node is None:
                    break
                path.append(next_node)
                curr = next_node
            if len(path) > 1:
                paths.append(path)
                
        # Check if all edges covered
        if len(used_edges) != len(edges):
            # Some cycles might remain if we didn't break all of them
            # This can happen if we only picked one start per cycle but there are multiple cycles
            # Or if the LP solution had cycles that we didn't start from.
            # Let's find remaining edges and form paths from them
            remaining_edges = set(edges) - set((min(u,v), max(u,v)) for u,v in used_edges)
            # This logic is getting complicated. 
            # Simpler: The LP ensures flow conservation. 
            # If we start from all sources, we cover all non-cycle edges.
            # Cycles must be broken manually.
            
            # Let's identify cycles in the remaining directed graph (edges not used)
            # Actually, used_edges tracks directed edges.
            rem_dir_edges = set(directed_edges) - used_edges
            if rem_dir_edges:
                # Decompose rem_dir_edges into paths
                # Build local adj
                rem_out = defaultdict(list)
                rem_in = defaultdict(list)
                for u, v in rem_dir_edges:
                    rem_out[u].append(v)
                    rem_in[v].append(u)
                    
                # Find starts in this subgraph
                rem_starts = []
                for u in rem_out:
                    if len(rem_out[u]) > len(rem_in[u]):
                        rem_starts.extend([u] * (len(rem_out[u]) - len(rem_in[u])))
                
                if not rem_starts:
                    # All cycles
                    # Pick arbitrary start for each cycle
                    visited = set()
                    for u in rem_out:
                        if u not in visited and rem_out[u]:
                            curr = u
                            path = [curr]
                            visited.add(curr)
                            while rem_out[curr]:
                                # Pick one
                                # Need to track used edges in this subgraph too
                                # This is messy. Let's just pick one neighbor.
                                nxt = rem_out[curr][0]
                                # Remove edge
                                rem_out[curr].remove(nxt)
                                rem_in[nxt].remove(curr)
                                curr = nxt
                                path.append(curr)
                                visited.add(curr)
                            if len(path) > 1:
                                paths.append(path)
                else:
                    used_rem = set()
                    for start in rem_starts:
                        curr = start
                        path = [curr]
                        while rem_out[curr]:
                            nxt = rem_out[curr][0]
                            rem_out[curr].remove(nxt)
                            rem_in[nxt].remove(curr)
                            used_rem.add((curr, nxt))
                            path.append(nxt)
                            curr = nxt
                        if len(path) > 1:
                            paths.append(path)
                            
        return paths

    def greedy_init(edges, seed_offset=0):
        rng = random.Random(seed_offset)
        remaining = set(edges)
        paths = []
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        while remaining:
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            if not deg: break
            candidates = sorted(deg.keys(), key=lambda x: (-deg[x], rng.random()))
            start = candidates[0]
            
            path = [start]
            pverts = {start}
            pedges = set()
            
            while True:
                best_next = None
                best_score = float('inf')
                nbrs = list(adj[start])
                rng.shuffle(nbrs)
                for v in nbrs:
                    e = (min(start, v), max(start, v))
                    if e in remaining and v not in pverts:
                        moves = 0
                        for w in adj[v]:
                            e2 = (min(v, w), max(v, w))
                            if e2 in remaining and w not in pverts and w != start:
                                moves += 1
                        score = moves + rng.random() * 0.1
                        if score < best_score:
                            best_score = score
                            best_next = v
                if best_next is None:
                    break
                path.append(best_next)
                pverts.add(best_next)
                e = (min(start, best_next), max(start, best_next))
                pedges.add(e)
                start = best_next
                
            if len(path) > 1:
                paths.append(path)
                for e in pedges:
                    remaining.discard(e)
            else:
                if remaining:
                    e = next(iter(remaining))
                    paths.append(list(e))
                    remaining.discard(e)
        return paths

    def try_merge(paths):
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                p1 = paths[i]
                p2 = paths[j]
                if len(p1) < 2 or len(p2) < 2:
                    continue
                set1 = set(p1)
                set2 = set(p2)
                common = set1 & set2
                if len(common) != 1:
                    continue
                shared = list(common)[0]
                is_ep1 = (p1[0] == shared or p1[-1] == shared)
                is_ep2 = (p2[0] == shared or p2[-1] == shared)
                if is_ep1 and is_ep2:
                    if p1[-1] == shared and p2[0] == shared:
                        new_p = p1 + p2[1:]
                    elif p1[0] == shared and p2[-1] == shared:
                        new_p = p2 + p1[1:]
                    elif p1[0] == shared and p2[0] == shared:
                        new_p = p1[::-1] + p2[1:]
                    elif p1[-1] == shared and p2[-1] == shared:
                        new_p = p2 + p1[::-1][1:]
                    else:
                        continue
                    if len(new_p) == len(set(new_p)):
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_p)
                        return new_paths
        return None

    def local_search(paths, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_paths = try_merge(paths)
            if new_paths is not None:
                paths = new_paths
                changed = True
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        best_paths = None
        best_count = float('inf')
        
        # Try MCF first
        if time.time() < DEADLINE - 5:
            mcf_paths = decompose_mcf(edges, DEADLINE)
            if mcf_paths is not None:
                # Verify validity
                valid = True
                edge_set = set(edges)
                used = set()
                for p in mcf_paths:
                    for i in range(len(p)-1):
                        e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                        if e not in edge_set or e in used:
                            valid = False
                            break
                        used.add(e)
                    if not valid: break
                if valid and used == edge_set:
                    best_paths = mcf_paths
                    best_count = len(mcf_paths)
        
        # If MCF failed or didn't improve, try heuristic
        if best_paths is None or time.time() < DEADLINE - 2:
             # Run heuristic to potentially improve or fallback
             for attempt in range(5):
                 if time.time() > DEADLINE - 2:
                     break
                 paths = greedy_init(edges, attempt)
                 paths = local_search(paths, DEADLINE - 2)
                 if len(paths) < best_count:
                     best_count = len(paths)
                     best_paths = paths
                     
        if best_paths is None:
            best_paths = greedy_init(edges, 0)
            
        result.append(best_paths)
        
    return result

# EVOLVE-BLOCK-END

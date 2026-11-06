"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles(edges, n, max_len=8, seed=0):
    """
    Find a set of candidate cycles up to max_len.
    Returns a list of cycles, where each cycle is a list of vertices.
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    cycles = []
    edge_set = set(edges)
    
    # Use DFS to find cycles
    # To avoid duplicates, we enforce that the smallest vertex in the cycle is the start
    # and we only traverse vertices > start until closing.
    
    start_time = time.time()
    time_limit = 2.0 # Limit time for cycle generation per graph
    
    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        # Only start from start if no smaller vertex is connected in a way that would form a cycle
        # Simplified: Just run DFS from each node, but prune paths that contain a node < start
        # This ensures each cycle is found exactly once (rooted at min vertex).
        
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            if len(path) >= 3 and start in adj[node]:
                # Check if edge (node, start) exists
                e = (min(node, start), max(node, start))
                if e in edge_set:
                    # Found a cycle
                    # Canonicalize: rotate so min is first, then sort by second vertex?
                    # Since we enforced min is start, and we only visit nodes > start,
                    # the path is unique for this start.
                    # But we might find the same cycle in reverse?
                    # DFS explores neighbors. To avoid reverse duplicates, 
                    # we can enforce that the second vertex in the path is < the last vertex?
                    # Or just add to a set of frozensets of edges.
                    cycle_edges = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                    # We'll store edge sets to deduplicate
                    cycles.append((path, cycle_edges))
            
            if len(path) < max_len:
                for neighbor in adj[node]:
                    if neighbor > start and neighbor not in visited:
                        # Prune: if neighbor is start and len(path)<3, skip
                        if neighbor == start:
                            continue
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))
    
    # Deduplicate cycles by edge set
    unique_cycles = {}
    for path, c_edges in cycles:
        unique_cycles[c_edges] = path
        
    return list(unique_cycles.values())

def _decompose(edges, n, seed, time_limit=10.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    start_time = time.time()
    
    # 1. Generate candidate cycles
    candidates = _find_cycles(edges, n, max_len=8, seed=seed)
    
    # If no cycles found, return all edges
    if not candidates:
        return [[u, v] for u, v in edges]
    
    # 2. Setup ILP
    # Variables: x_i for each candidate cycle i
    m = len(candidates)
    
    # Objective: Maximize sum (len(c) - 1) * x_c
    # Minimize sum (1 - (len(c) - 1)) * x_c = sum (2 - len(c)) * x_c
    # Or just minimize -sum (len(c) - 1) * x_c
    c_obj = np.array([-(len(c) - 1) for c in candidates], dtype=float)
    
    # Constraints: For each edge, sum of x_c for c containing edge <= 1
    # Build incidence matrix A (m_edges x m_cycles)
    edge_index = {e: i for i, e in enumerate(edges)}
    m_edges = len(edges)
    
    # Sparse matrix for efficiency
    # scipy.optimize.milp accepts LinearConstraint with sparse A
    import scipy.sparse as sp
    
    rows = []
    cols = []
    vals = []
    
    for i, cycle in enumerate(candidates):
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in edge_index:
                rows.append(edge_index[e])
                cols.append(i)
                vals.append(1.0)
    
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m_edges, m))
    
    # Constraints: A @ x <= 1
    lb = np.zeros(m_edges)
    ub = np.ones(m_edges)
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(m) # All binary
    bounds = Bounds(0, 1)
    
    # Solve ILP
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success and res.x is not None:
            selected_cycles = [candidates[i] for i in range(m) if res.x[i] > 0.5]
            
            # Verify validity and construct pieces
            used_edges = set()
            pieces = []
            for cycle in selected_cycles:
                c_edges = set()
                valid = True
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges or e not in edge_index:
                        valid = False
                        break
                    c_edges.add(e)
                
                if valid:
                    used_edges.update(c_edges)
                    pieces.append(cycle)
            
            # Add remaining edges as single pieces
            for e in edges:
                if e not in used_edges:
                    pieces.append([e[0], e[1]])
            
            return pieces
    except Exception:
        pass
    
    # Fallback: Greedy heuristic if ILP fails or is too slow
    # This is the old logic, kept as backup
    best_pieces = None
    best_count = len(edges)
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        remaining = set(edges)
        pieces = []
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)
            
        rng = random.Random(seed + restart)
        
        while remaining:
            candidates_v = [(len(adj[u]), u) for u in adj if adj[u]]
            if not candidates_v:
                break
            candidates_v.sort(reverse=True)
            top_k = min(5, len(candidates_v))
            _, start = rng.choice(candidates_v[:top_k])
            
            found_cycle = None
            visited = set()
            path = []
            
            def dfs(u, depth):
                nonlocal found_cycle, visited, path
                if found_cycle is not None:
                    return True
                if depth > n + 5:
                    return False
                
                visited.add(u)
                path.append(u)
                
                neighbors = []
                for v in adj[u]:
                    e1 = (min(u, v), max(u, v))
                    if e1 in remaining:
                        neighbors.append(v)
                
                rng.shuffle(neighbors)
                
                for v in neighbors:
                    if v == start and len(path) >= 3:
                        found_cycle = list(path)
                        return True
                    if v not in visited:
                        if dfs(v, depth + 1):
                            return True
                
                path.pop()
                visited.discard(u)
                return False
            
            dfs(start, 0)
            
            if found_cycle:
                cycle_edges = set()
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        if v in adj[u]: adj[u].remove(v)
                        if u in adj[v]: adj[v].remove(u)
                
                pieces.append(found_cycle)
            else:
                best_edge_node = None
                best_deg = -1
                for u in list(adj.keys()):
                    if adj[u]:
                        deg = len(adj[u])
                        if deg > best_deg:
                            best_deg = deg
                            best_edge_node = u
                
                if best_edge_node is not None and adj[best_edge_node]:
                    v = rng.choice(adj[best_edge_node])
                    e = (min(best_edge_node, v), max(best_edge_node, v))
                    if e in remaining:
                        remaining.discard(e)
                        if v in adj[best_edge_node]: adj[best_edge_node].remove(v)
                        if best_edge_node in adj[v]: adj[v].remove(best_edge_node)
                        pieces.append([best_edge_node, v])
                else:
                    e = next(iter(remaining))
                    remaining.discard(e)
                    u, v = e
                    if v in adj[u]: adj[u].remove(v)
                    if u in adj[v]: adj[v].remove(u)
                    pieces.append([u, v])
        
        current_count = len(pieces)
        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Total time budget 150s. Safety margin 10s. 140s for work.
    # 3 graphs. ~45s each.
    time_per_graph = 45.0 
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=time_per_graph)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END

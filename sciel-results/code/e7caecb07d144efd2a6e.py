"""K2,3-free graph on 20 vertices: maximize edges via greedy + local search."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy 1: Structured warm-start
    if time.time() < deadline - 5.0:
        structured = _structured_warmstart(n)
        if structured:
            improved = _local_search(n, structured, deadline)
            if len(improved) > best_score:
                best_score = len(improved)
                best_edges = improved[:]
    
    # Strategy 2: Multiple greedy restarts
    for seed in range(300):
        if time.time() > deadline - 10.0:
            break
        
        edges = _greedy_construct(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 3: Local search from best greedy result
    if time.time() < deadline - 5.0 and best_edges:
        improved = _local_search(n, best_edges, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved[:]
    
    # Strategy 4: More greedy restarts with heuristics + local search
    for seed in range(300, 800):
        if time.time() > deadline - 5.0:
            break
        
        edges = _greedy_construct_heuristic(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
        
        if score >= best_score - 3 and time.time() < deadline - 5.0:
            improved = _local_search(n, edges, deadline)
            if len(improved) > best_score:
                best_score = len(improved)
                best_edges = improved[:]
    
    # Strategy 5: Final polish
    if time.time() < deadline - 2.0 and best_edges:
        polished = _local_search(n, best_edges, deadline)
        if len(polished) > best_score:
            best_score = len(polished)
            best_edges = polished[:]
    
    return best_edges


def _structured_warmstart(n):
    """Create a structured K2,3-free graph using cyclic/regular construction."""
    best_circ = []
    best_circ_score = 0
    
    # Try circulant graphs with various connection sets
    candidate_sets = [
        {1, 2, 5, 7},
        {1, 2, 4, 8},
        {1, 3, 5, 9},
        {1, 2, 6, 8},
        {1, 3, 6, 9},
        {1, 2, 4, 6},
        {1, 2, 3, 7},
        {1, 2, 5, 9},
        {1, 3, 4, 8},
        {1, 3, 5, 7},
        {1, 2, 5, 6},
        {1, 3, 4, 7},
    ]
    
    for conn_set in candidate_sets:
        adj = [set() for _ in range(n)]
        edges = []
        valid = True
        
        for u in range(n):
            for s in sorted(conn_set):
                v = (u + s) % n
                if v > u:
                    if v not in adj[u]:
                        if _can_add_edge(adj, u, v, n):
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.append([u, v])
                        else:
                            valid = False
                            break
            if not valid:
                break
        
        if valid and len(edges) > best_circ_score:
            best_circ_score = len(edges)
            best_circ = edges[:]
    
    # Also try group-based construction
    adj = [set() for _ in range(n)]
    edges = []
    
    # Intra-group edges (K4s in groups of 4)
    for g in range(5):
        group = [g*4, g*4+1, g*4+2, g*4+3]
        for i in range(4):
            for j in range(i+1, 4):
                u, v = group[i], group[j]
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Add inter-group edges
    for g1 in range(5):
        for g2 in range(g1+1, 5):
            for offset in range(4):
                for i in range(4):
                    u = g1*4 + i
                    v = g2*4 + ((i + offset) % 4)
                    if _can_add_edge(adj, u, v, n):
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
    
    if len(edges) > best_circ_score:
        best_circ = edges[:]
        best_circ_score = len(edges)
    
    return best_circ


def _greedy_construct(n, seed):
    """Greedy construction: add edges in random order if K2,3-free."""
    random.seed(seed)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in all_edges:
        if _can_add_edge(adj, u, v, n):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges


def _greedy_construct_heuristic(n, seed):
    """Greedy with heuristic: prefer edges between low-degree vertices."""
    random.seed(seed)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    adj = [set() for _ in range(n)]
    edges = []
    remaining = set(all_edges)
    
    while remaining and len(edges) < 190:
        best_edge = None
        best_score = -1
        
        candidates = list(remaining)
        if len(candidates) > 150:
            candidates = random.sample(candidates, 150)
        
        for u, v in candidates:
            if not _can_add_edge(adj, u, v, n):
                continue
            
            deg_sum = len(adj[u]) + len(adj[v])
            common = len(adj[u] & adj[v])
            score = -(deg_sum * 2 + common * 5)
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        remaining.discard((u, v))
    
    return edges


def _can_add_edge(adj, u, v, n):
    """Check if adding edge (u,v) keeps the graph K2,3-free."""
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    for w in adj[v]:
        if w == u:
            continue
        common = adj[u] & adj[w]
        common = common | {v}
        if len(common) >= 3:
            return False
    
    for w in adj[u]:
        if w == v:
            continue
        common = adj[v] & adj[w]
        common = common | {u}
        if len(common) >= 3:
            return False
    
    return True


def _local_search(n, edges, deadline):
    """Local search: add missing edges, then try 1-edge and 2-edge swaps."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_edges = [e[:] for e in edges]
    
    # Phase 1: Add all possible missing edges
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
    random.shuffle(missing)
    
    added = True
    while added and time.time() < deadline - 3.0:
        added = False
        for u, v in missing:
            if v in adj[u]:
                continue
            if _can_add_edge(adj, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                added = True
    
    # Phase 2: 1-edge swap moves
    for iteration in range(150):
        if time.time() > deadline - 3.0:
            break
        if not current_edges:
            break
        
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove edge
        adj[u].discard(v)
        adj[v].discard(u)
        current_edges.pop(idx)
        
        # Try to add all missing edges
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in adj[a]]
        random.shuffle(missing2)
        
        new_edges_added = []
        for a, b in missing2:
            if _can_add_edge(adj, a, b, n):
                adj[a].add(b)
                adj[b].add(a)
                new_edges_added.append([a, b])
        
        # Keep the change if we gained at least 1 net edge
        if len(new_edges_added) >= 2:
            current_edges.extend(new_edges_added)
        else:
            # Revert: remove newly added, restore original
            for a, b in new_edges_added:
                adj[a].discard(b)
                adj[b].discard(a)
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append([u, v])
    
    # Phase 3: 2-edge swap moves
    for iteration in range(100):
        if time.time() > deadline - 3.0:
            break
        if len(current_edges) < 2:
            break
        
        # Pick 2 random edges
        idx1 = random.randint(0, len(current_edges) - 1)
        idx2 = random.randint(0, len(current_edges) - 1)
        if idx2 > idx1:
            idx2 -= 1
        
        e1 = current_edges[idx1][:]
        e2 = current_edges[idx2][:]
        
        # Remove both edges
        u1, v1 = e1
        u2, v2 = e2
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        current_edges.pop(idx1)
        current_edges.pop(idx2)
        
        # Try to add all missing edges
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in adj[a]]
        random.shuffle(missing2)
        
        new_edges_added = []
        for a, b in missing2:
            if _can_add_edge(adj, a, b, n):
                adj[a].add(b)
                adj[b].add(a)
                new_edges_added.append([a, b])
        
        # Keep if we gained at least 1 net edge (added 3+)
        if len(new_edges_added) >= 3:
            current_edges.extend(new_edges_added)
        else:
            # Revert
            for a, b in new_edges_added:
                adj[a].discard(b)
                adj[b].discard(a)
            adj[u1].add(v1)
            adj[v1].add(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            current_edges.append(e1)
            current_edges.append(e2)
    
    return current_edges


# EVOLVE-BLOCK-END


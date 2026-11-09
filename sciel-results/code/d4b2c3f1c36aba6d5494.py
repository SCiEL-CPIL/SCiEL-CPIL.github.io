"""
K4-free graph without independent 6-set, maximizing n.
Uses a constructive approach based on known Ramsey graphs.
"""
import time
import random

def _has_k4(adj, n):
    """Check if graph has a K4. adj is list of sets."""
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                continue
            for k in range(j + 1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                common = adj[i] & adj[j] & adj[k]
                for l in common:
                    if l > k:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    comp = [set(range(n)) - {i} - adj[i] for i in range(n)]
    
    def find_clique(vertices, size):
        if size == 0:
            return True
        if not vertices:
            return False
        if len(vertices) < size:
            return False
        v_list = list(vertices)
        for idx, v in enumerate(v_list):
            if len(v_list) - idx < size:
                break
            new_vertices = vertices - {v} & comp[v]
            if find_clique(new_vertices, size - 1):
                return True
            vertices = vertices - {v}
        return False
    
    return find_clique(set(range(n)), 6)

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_independent_6(adj, n)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _paley_graph_17():
    """Return edges for Paley graph of order 17."""
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return edges

def _try_grow(adj, n, max_grow=5, deadline=None):
    """Try to grow the graph by adding vertices."""
    for new_n in range(n + 1, n + max_grow + 1):
        if deadline and time.time() > deadline - 5:
            break
        # Try to add vertex new_n with some edges to existing vertices
        best_addition = None
        best_score = -1
        for attempt in range(100):
            # Randomly choose edges for new vertex
            adj_new = [set(s) for s in adj]
            adj_new.append(set())
            # Add some edges to new vertex
            for i in range(new_n):
                if random.random() < 0.3:
                    adj_new[i].add(new_n)
                    adj_new[new_n].add(i)
            if _is_valid(adj_new, new_n):
                # Count how many edges we have - prefer more edges (more constraints)
                score = sum(len(s) for s in adj_new)
                if score > best_score:
                    best_score = score
                    best_addition = adj_new
        if best_addition is not None:
            return best_addition, new_n
    return adj, n

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_edges = None
    
    # Start with Paley graph of order 17
    edges_17 = _paley_graph_17()
    adj_17 = _build_adj(edges_17, 17)
    if _is_valid(adj_17, 17):
        best_n = 17
        best_edges = edges_17
    
    # Try to grow from 17
    for attempt_seed in range(50):
        if time.time() > deadline - 30:
            break
        random.seed(attempt_seed)
        adj_grown, n_grown = _try_grow(adj_17, 17, max_grow=15, deadline=deadline)
        if n_grown > best_n:
            best_n = n_grown
            best_edges = [[i, j] for i in range(n_grown) for j in range(i + 1, n_grown) if j in adj_grown[i]]
    
    # Try local search from best found
    if best_n >= 17 and time.time() < deadline - 20:
        # Do a simple local search to try to improve
        adj_current = _build_adj(best_edges, best_n)
        for step in range(1000):
            if time.time() > deadline - 10:
                break
            # Try to add a random edge
            u, v = random.sample(range(best_n), 2)
            if v not in adj_current[u]:
                adj_current[u].add(v)
                adj_current[v].add(u)
                if not _is_valid(adj_current, best_n):
                    adj_current[u].remove(v)
                    adj_current[v].remove(u)
    
    # Final verification
    if best_edges is not None and best_n > 0:
        adj_final = _build_adj(best_edges, best_n)
        if not _is_valid(adj_final, best_n):
            # Fallback to Paley 17
            best_n = 17
            best_edges = _paley_graph_17()
    
    return {'n': best_n, 'edges': best_edges}

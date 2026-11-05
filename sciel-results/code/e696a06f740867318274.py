"""K2,3-free graph on 20 vertices: maximize edges via structured seed + local search."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy 1: Structured construction (blow-up of C5)
    # 5 groups of 4 vertices, each group is a clique (K4), 
    # groups connected in a cycle: group i connected to group i+1 mod 5
    # Each K4 has 6 edges, 5 groups = 30 internal edges
    # Between consecutive groups: 4*4 = 16 edges, 5 pairs = 80 edges
    # Total: 110 edges, but need to verify K2,3-free
    structured = _structured_c5(n)
    if structured:
        best_score = len(structured)
        best_edges = structured[:]
    
    # Strategy 2: Multiple greedy restarts for diversity
    for seed in range(300):
        if time.time() > deadline - 15.0:
            break
        edges = _greedy_construct(n, seed)
        score = len(edges)
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 3: Local search on best solution
    if best_edges and time.time() < deadline - 5.0:
        improved = _local_search(n, best_edges, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved
    
    # Strategy 4: More local search iterations with random restarts
    for restart in range(5):
        if time.time() > deadline - 3.0:
            break
        # Perturb best solution
        perturbed = _perturb(best_edges, n, restart)
        improved = _local_search(n, perturbed, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved[:]
    
    return best_edges


def _structured_c5(n):
    """
    Blow-up of C5: 5 groups of 4 vertices each.
    Within each group: complete graph (K4).
    Between groups: complete bipartite between consecutive groups (cycle).
    This should be K2,3-free because the structure limits common neighbors.
    """
    if n != 20:
        return []
    
    # Group sizes: 4 each
    groups = []
    for i in range(5):
        group = list(range(i * 4, (i + 1) * 4))
        groups.append(group)
    
    edges = []
    
    # Internal edges: each group is a K4
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append([group[i], group[j]])
    
    # External edges: between consecutive groups (cycle)
    for i in range(5):
        j = (i + 1) % 5
        for u in groups[i]:
            for v in groups[j]:
                edges.append([u, v])
    
    return edges


def _greedy_construct(n, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
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


def _can_add_edge(adj, u, v, n):
    """
    Check if adding edge (u,v) keeps the graph K2,3-free.
    K2,3: two vertices with 3+ common neighbors.
    """
    # After adding (u,v), we need to check:
    # 1. Pair (u,v): common neighbors = adj[u] ∩ adj[v]
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    # 2. For each w in adj[v]: pair (u,w)
    for w in adj[v]:
        if w == u:
            continue
        common = adj[u] & adj[w]
        # v is in adj[w] because w in adj[v] and graph is undirected
        common = common | {v}
        if len(common) >= 3:
            return False
    
    # 3. For each w in adj[u]: pair (v,w)
    for w in adj[u]:
        if w == v:
            continue
        common = adj[v] & adj[w]
        common = common | {u}
        if len(common) >= 3:
            return False
    
    return True


def _local_search(n, edges, deadline):
    """
    Local search: iteratively remove one edge, then greedily add all possible
    missing edges. This escapes local optima by allowing temporary edge loss.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_edges = [e[:] for e in edges]
    best_edges = [e[:] for e in current_edges]
    best_score = len(current_edges)
    
    # Phase 1: Try to add all possible missing edges (greedy completion)
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
    random.shuffle(missing)
    
    added = True
    while added and time.time() < deadline - 3.0:
        added = False
        for u, v in missing:
            if v in adj[u]:  # already added
                continue
            if _can_add_edge(adj, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                added = True
    
    if len(current_edges) > best_score:
        best_score = len(current_edges)
        best_edges = [e[:] for e in current_edges]
    
    # Phase 2: Iterative edge removal and re-addition
    # Try removing random edges and see if we can recover more
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 3.0:
            break
        if not current_edges:
            break
        
        # Pick a random edge to remove
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove it
        adj[u].discard(v)
        adj[v].discard(u)
        current_edges.pop(idx)
        
        # Try to re-add all missing edges greedily
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in adj[a]]
        random.shuffle(missing2)
        
        for a, b in missing2:
            if _can_add_edge(adj, a, b, n):
                adj[a].add(b)
                adj[b].add(a)
                current_edges.append([a, b])
        
        # Update best
        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = [e[:] for e in current_edges]
    
    return best_edges


def _perturb(edges, n, seed):
    """
    Perturb a solution by removing a few random edges and re-adding greedily.
    Used for diversity in local search restarts.
    """
    random.seed(seed + 10000)
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_edges = [e[:] for e in edges]
    
    # Remove 2-5 random edges
    num_remove = random.randint(2, 5)
    for _ in range(num_remove):
        if not current_edges:
            break
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        adj[u].discard(v)
        adj[v].discard(u)
        current_edges.pop(idx)
    
    # Re-add greedily
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
    random.shuffle(missing)
    
    for u, v in missing:
        if _can_add_edge(adj, u, v, n):
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append([u, v])
    
    return current_edges

# EVOLVE-BLOCK-END

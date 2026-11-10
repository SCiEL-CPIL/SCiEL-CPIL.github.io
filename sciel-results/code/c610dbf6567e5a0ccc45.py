"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _adj_to_bitmasks(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    """Check if graph contains a K4 using bitmask operations."""
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j+1, n):
                if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                    continue
                # i,j,k form a triangle; check if any vertex connects to all three
                common = adj[i] & adj[j] & adj[k]
                # Remove i, j, k themselves
                common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                if common:
                    return True
    return False

def _max_triangle_free_size(adj, n, deadline=None):
    """Branch and bound to find the largest triangle-free induced subgraph."""
    # Use bitmask DP / branch and bound
    # State: (current_vertex_index, current_selected_bitmask, current_size)
    # We want to maximize size such that induced subgraph is triangle-free
    
    # A triangle-free graph: for every edge in the induced subgraph, the two endpoints share no common neighbor in the subgraph.
    # Equivalently: no three vertices form a triangle.
    
    best = [0]
    
    # We'll use a recursive approach with pruning
    # Key insight: if we process vertices in order, we can track which vertices are "forbidden"
    # (cannot be added because they'd form a triangle with two already-selected vertices)
    
    def dfs(start, selected, size, forbidden):
        if deadline and time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        # Pruning: even if we add all remaining vertices, can we beat best?
        remaining = n - start
        if size + remaining <= best[0]:
            return
        for v in range(start, n):
            if selected & (1 << v):
                continue
            if forbidden & (1 << v):
                continue
            # Check if adding v creates a triangle with any two already selected vertices
            # v must not form a triangle: for any edge (u,w) in selected, v cannot be adjacent to both u and w
            # More efficiently: v's neighbors in selected must form an independent set in the original graph
            # i.e., no two neighbors of v in selected are adjacent to each other
            neighbors_in_selected = adj[v] & selected
            # Check if neighbors_in_selected contains an edge
            has_edge = False
            temp = neighbors_in_selected
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                temp -= lsb
                if adj[u] & neighbors_in_selected & ~lsb:
                    has_edge = True
                    break
            if has_edge:
                continue
            # Add v
            new_selected = selected | (1 << v)
            # New forbidden: vertices that would form a triangle with v and some selected vertex
            # A vertex x is forbidden if there exists y in selected such that x-y and x-v are edges, 
            # meaning x is adjacent to both v and some y in selected, AND y is adjacent to v
            # Actually: x is forbidden if x is adjacent to v AND x is adjacent to some y in selected 
            # where y is also adjacent to v (so x,v,y form a triangle)
            # So forbidden_new = (adj[v] & adj[neighbors_in_selected]) ... 
            # More precisely: for each y in selected that is adjacent to v, any x adjacent to both v and y is forbidden
            new_forbidden = forbidden
            for y_bit in range(n):
                if (selected >> y_bit) & 1 and (adj[v] >> y_bit) & 1:
                    new_forbidden |= adj[y_bit] & adj[v]
            # Also, v itself should not be in forbidden for future, but we're moving past it
            new_forbidden &= ~(1 << v)  # v is now selected, not forbidden
            dfs(v + 1, new_selected, size + 1, new_forbidden)
    
    dfs(0, 0, 0, 0)
    return best[0]

def _evaluate_graph(edges, n=18):
    """Evaluate a graph: return (is_k4_free, max_tf_size)."""
    adj = _adj_to_bitmasks(edges, n)
    if _has_k4(adj, n):
        return False, 999
    max_tf = _max_triangle_free_size(adj, n)
    return True, max_tf

def construct():
    n = 18
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with complete tripartite K6,6,6 as baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    base_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    base_edges.append([a, b])
    
    best_edges = base_edges
    best_score = 12  # K6,6,6 has max triangle-free induced subgraph of 12
    
    # Try some hand-crafted constructions first
    # Try K9,9 minus some edges to break triangle-free property of the whole graph
    # Actually, let's try: take a random K4-free graph and do local search
    
    random.seed(42)
    
    # Local search: start from K6,6,6, try adding/removing edges
    # We need to maintain K4-free property
    
    def get_edge_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u,v), max(u,v)))
        return es
    
    def edges_to_set(edges):
        return get_edge_set(edges)
    
    current_edges = list(base_edges)
    current_set = edges_to_set(current_edges)
    
    # Precompute adj for fast K4 check
    current_adj = _adj_to_bitmasks(current_edges, n)
    
    # Try to improve by removing edges from within parts (making parts not complete)
    # and adding edges between parts carefully
    
    # Actually, let's try a different strategy: use simulated annealing / local search
    # on the edge set, maintaining K4-free constraint
    
    # First, let's try some structured candidates
    
    candidates = [base_edges]
    
    # Try: split into 3 groups but make them not complete tripartite
    # e.g., K5,6,7 with some internal edges
    for split in [(5,6,7), (4,7,7), (3,8,7), (6,5,7), (5,7,6)]:
        if sum(split) != 18:
            continue
        parts_c = []
        start = 0
        for s in split:
            parts_c.append(list(range(start, start+s)))
            start += s
        edges_c = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts_c[i]:
                    for b in parts_c[j]:
                        edges_c.append([a, b])
        candidates.append(edges_c)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        k4_free, max_tf = _evaluate_graph(cand, n)
        if k4_free and max_tf < best_score:
            best_score = max_tf
            best_edges = list(cand)
    
    # Local search from best candidate
    current_edges = list(best_edges)
    current_set = edges_to_set(current_edges)
    current_adj = _adj_to_bitmasks(current_edges, n)
    current_max_tf = best_score
    
    # Local search: try single edge additions/removals
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for iteration in range(5000):
        if time.time() > deadline - 20:
            break
        
        improved = False
        # Try random edge removals
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges)-1)
            u, v = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            new_set = edges_to_set(new_edges)
            new_adj = _adj_to_bitmasks(new_edges, n)
            if _has_k4(new_adj, n):
                continue
            max_tf = _max_triangle_free_size(new_adj, n)
            if max_tf < current_max_tf:
                current_edges = new_edges
                current_set = new_set
                current_adj = new_adj
                current_max_tf = max_tf
                if max_tf < best_score:
                    best_score = max_tf
                    best_edges = list(current_edges)
                improved = True
                break
            elif max_tf == current_max_tf and random.random() < 0.1:
                # Accept equal with some probability for exploration
                current_edges = new_edges
                current_set = new_set
                current_adj = new_adj
                improved = True
                break
        
        # Try random edge additions
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            u, v = random.choice(all_possible_edges)
            if (u, v) in current_set:
                continue
            new_edges = current_edges + [[u, v]]
            new_set = edges_to_set(new_edges)
            new_adj = _adj_to_bitmasks(new_edges, n)
            if _has_k4(new_adj, n):
                continue
            max_tf = _max_triangle_free_size(new_adj, n)
            if max_tf < current_max_tf:
                current_edges = new_edges
                current_set = new_set
                current_adj = new_adj
                current_max_tf = max_tf
                if max_tf < best_score:
                    best_score = max_tf
                    best_edges = list(current_edges)
                improved = True
                break
        
        if not improved and iteration > 100:
            # Try a restart with a different random seed
            random.seed(iteration)
    
    return best_edges
# EVOLVE-BLOCK-END

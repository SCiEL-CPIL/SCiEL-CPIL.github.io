import numpy as np
import random
import time

N = 31
E = 91
DEADLINE_SECONDS = 118.0  # safety margin under the 150 s limit (evaluator kills at 150s)

# Precompute all possible edges
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
TOTAL_EDGES = len(ALL_EDGES)  # 465

# Precompute edge index for fast lookup
EDGE_INDEX = {e: i for i, e in enumerate(ALL_EDGES)}

def _count_c4_from_adj(adj):
    """
    Count C4s in graph represented by adjacency matrix.
    C4 count = 1/2 * sum_{i<j} C(A2[i,j], 2)
    where A2 = adj @ adj.
    """
    A2 = adj @ adj
    i, j = np.triu_indices(N, k=1)
    common_counts = A2[i, j]
    terms = common_counts * (common_counts - 1) // 2
    total = np.sum(terms)
    return total // 2

def _delta_c4_swap(adj, remove_edge, add_edge):
    """
    Compute the change in C4 count when removing remove_edge and adding add_edge.
    Uses the fact that C4 count = 1/2 * sum_{i<j} C(d_ij, 2) where d_ij = number of common neighbors.
    
    When we remove edge (u,v): d_ij changes for pairs involving u or v.
    When we add edge (x,y): d_ij changes for pairs involving x or y.
    
    This is complex to compute exactly without recomputing. For a 31-vertex graph,
    recomputing the full C4 count is fast enough (~0.5ms). So we just recompute.
    """
    # For N=31, full recomputation is fast. We'll just recompute.
    pass

def _evaluate_swap(adj, edge_set, edges_list, remove_idx, add_edge):
    """
    Perform a swap: remove edges_list[remove_idx], add add_edge.
    Return new_c4 and whether the swap was valid.
    """
    u1, v1 = edges_list[remove_idx]
    u2, v2 = add_edge
    
    # Check if add_edge is already in the graph
    if add_edge in edge_set:
        return None, False
    
    # Perform swap
    adj[u1, v1] = 0
    adj[v1, u1] = 0
    edge_set.remove((u1, v1))
    
    adj[u2, v2] = 1
    adj[v2, u2] = 1
    edge_set.add((u2, v2))
    
    new_c4 = _count_c4_from_adj(adj)
    
    # Revert
    adj[u2, v2] = 0
    adj[v2, u2] = 0
    edge_set.remove((u2, v2))
    
    adj[u1, v1] = 1
    adj[v1, u1] = 1
    edge_set.add((u1, v1))
    
    return new_c4, True

def _local_search(adj, edge_set, edges_list, start_time, deadline, best_score, best_edges):
    """
    Local search with simulated annealing-like acceptance.
    Returns updated best_score and best_edges.
    """
    current_c4 = _count_c4_from_adj(adj)
    current_edges = edges_list[:]
    current_edge_set = edge_set.copy()
    current_adj = adj.copy()
    
    # Track the best in this local search
    local_best_c4 = current_c4
    local_best_edges = current_edges[:]
    
    # Simulated annealing parameters
    T = 10.0  # initial temperature
    cooling = 0.95
    
    while time.time() < deadline - 3:
        # Pick a random edge to remove
        remove_idx = random.randint(0, E - 1)
        u1, v1 = current_edges[remove_idx]
        
        # Pick a random non-edge to add
        # Try up to 10 times to find a valid non-edge
        valid = False
        for _ in range(10):
            u2, v2 = random.choice(ALL_EDGES)
            if (u2, v2) not in current_edge_set:
                valid = True
                break
        
        if not valid:
            continue
        
        # Evaluate the swap
        new_c4, ok = _evaluate_swap(current_adj, current_edge_set, current_edges, remove_idx, (u2, v2))
        if not ok:
            continue
        
        delta = new_c4 - current_c4
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta <= 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
            # Accept the move
            current_adj[u1, v1] = 0
            current_adj[v1, u1] = 0
            current_edge_set.remove((u1, v1))
            
            current_adj[u2, v2] = 1
            current_adj[v2, u2] = 1
            current_edge_set.add((u2, v2))
            
            current_edges[remove_idx] = (u2, v2)
            current_c4 = new_c4
            
            if current_c4 < local_best_c4:
                local_best_c4 = current_c4
                local_best_edges = current_edges[:]
            
            # Update global best
            if current_c4 < best_score:
                best_score = current_c4
                best_edges = current_edges[:]
            
            # Cool down
            T *= cooling
        else:
            # Reject: no changes to current state
            T *= cooling
        
        if local_best_c4 <= 1:
            break
    
    return best_score, best_edges, local_best_c4, local_best_edges

def _greedy_build():
    """
    Build a graph greedily: start empty, add E edges one by one,
    choosing the edge that minimizes the increase in C4 count.
    """
    adj = np.zeros((N, N), dtype=int)
    edge_set = set()
    edges_list = []
    
    # Precompute a shuffled order of all edges
    shuffled = ALL_EDGES[:]
    random.shuffle(shuffled)
    
    for edge in shuffled:
        if len(edges_list) >= E:
            break
        u, v = edge
        # Check if adding this edge would create many C4s
        # New C4s = number of common neighbors of u and v that are already connected
        # Actually, new C4s through (u,v) = for each w in N(v)\{u}, x in N(u)\{v}, if w~x
        # But we can just compute the delta by temporarily adding and counting
        adj[u, v] = 1
        adj[v, u] = 1
        new_c4 = _count_c4_from_adj(adj)
        adj[u, v] = 0
        adj[v, u] = 0
        
        # For greedy, we just add it (this is a simple greedy, not optimal)
        # A better greedy would sort candidates by delta, but that's O(E^2) per step
        edge_set.add((u, v))
        edges_list.append((u, v))
        adj[u, v] = 1
        adj[v, u] = 1
    
    return adj, edge_set, edges_list

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_edges = None
    
    # Strategy 1: Greedy build + local search
    adj, edge_set, edges_list = _greedy_build()
    best_score, best_edges, _, _ = _local_search(
        adj, edge_set, edges_list, start_time, DEADLINE_SECONDS, best_score, best_edges
    )
    
    # Strategy 2: Multiple random restarts with local search
    # Use remaining time for more restarts
    while time.time() < DEADLINE_SECONDS - 10:
        # Random initialization
        edges = random.sample(ALL_EDGES, E)
        edge_set = set(edges)
        adj = np.zeros((N, N), dtype=int)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Local search with simulated annealing
        best_score, best_edges, local_best, _ = _local_search(
            adj, edge_set, edges, start_time, DEADLINE_SECONDS, best_score, best_edges
        )
        
        if best_score <= 1:
            break
    
    # Ensure best_edges is valid
    if best_edges is None:
        best_edges = random.sample(ALL_EDGES, E)
    
    return best_edges

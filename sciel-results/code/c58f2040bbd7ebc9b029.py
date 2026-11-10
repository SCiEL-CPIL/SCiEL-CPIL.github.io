# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Strategy:
      1. Start from a dense base (Paley-like or random)
      2. Repair any K2,3 violations
      3. Greedy add edges
      4. Local search (swap edges to increase count)
      5. Multiple restarts with deadline guard
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    n = 20
    
    def elapsed():
        return time.time() - start_time
    
    def is_k23_free(adj):
        """Check no pair of vertices has >=3 common neighbors."""
        # adj is n x n 0/1 matrix
        # common[u,v] = sum_w adj[u,w]*adj[v,w] for u!=v
        # Using matrix multiply: (adj @ adj)[u,v] gives common neighbors for u!=v (since adj[u,u]=0)
        common = adj @ adj
        for u in range(n):
            for v in range(u + 1, n):
                if common[u, v] >= 3:
                    return False
        return True
    
    def count_violations(adj):
        """Count number of pairs with >=3 common neighbors."""
        common = adj @ adj
        cnt = 0
        for u in range(n):
            for v in range(u + 1, n):
                if common[u, v] >= 3:
                    cnt += 1
        return cnt
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u, v]:
            return True
        # Adding (u,v) affects common neighbor counts for pairs involving u or v
        # For pair (u, w) where w != v: common neighbors increase by 1 if adj[v,w]=1
        # For pair (v, w) where w != u: common neighbors increase by 1 if adj[u,w]=1
        # For pair (u, v): common neighbors = sum_x adj[u,x]*adj[v,x]
        # We need to check all pairs that could now have >=3 common neighbors
        
        # Efficient check: for each w, if adj[u,w] and adj[v,w], then (u,v) common neighbor count increases
        # Actually, adding edge (u,v) changes:
        # - common[u,v] becomes sum_x adj[u,x]*adj[v,x] (was 0 since no edge, but common neighbors still counted)
        #   Wait, common neighbors of u and v don't depend on whether u-v edge exists!
        #   common[u,v] = |N(u) ∩ N(v)| which doesn't include u or v themselves.
        #   Adding edge (u,v) doesn't change N(u) ∩ N(v) directly, but:
        #   - u is now in N(v) and v is now in N(u)
        #   - So for pair (u, w): N(u) gains v, so common[u,w] gains 1 if w is in N(v) i.e. adj[v,w]=1
        #   - For pair (v, w): N(v) gains u, so common[v,w] gains 1 if w is in N(u) i.e. adj[u,w]=1
        #   - For pair (u, v): N(u) gains v, N(v) gains u, but common[u,v] = |N(u)∩N(v)| 
        #     The new common neighbors include any w where adj[u,w]=adj[v,w]=1, plus... 
        #     Actually u is not in N(u) and v is not in N(v), so common[u,v] = |N(u)∩N(v)| unchanged in value
        #     but wait, N(u) now includes v, N(v) now includes u. Does v ∈ N(u)∩N(v)? v ∈ N(u) yes, v ∈ N(v)? No.
        #     Does u ∈ N(u)∩N(v)? u ∈ N(u)? No. So common[u,v] unchanged.
        
        # So we need to check:
        # 1. For each w != u,v: if adj[u,w] and adj[v,w], then common[u,w] and common[v,w] each increase by 1
        #    Check if any such pair would now have >=3
        # 2. Also pairs (u,w) where adj[v,w]=1: common[u,w] increases by 1
        # 3. Pairs (v,w) where adj[u,w]=1: common[v,w] increases by 1
        
        # Get current common neighbor counts
        # common[u,w] for w != u: sum_x adj[u,x]*adj[w,x]
        # Let's just compute the affected pairs
        
        # Precompute adjacency as lists for speed
        N_u = set()
        N_v = set()
        for w in range(n):
            if adj[u, w] and w != u:
                N_u.add(w)
            if adj[v, w] and w != v:
                N_v.add(w)
        
        # After adding (u,v):
        # N_u_new = N_u ∪ {v}, N_v_new = N_v ∪ {u}
        
        # Check pair (u, w) for w in N_v (w != u): common[u,w] increases by 1 (because v added to N_u, and v ∈ N_w? no...)
        # Wait, I'm confusing myself. Let me think again.
        
        # common[u,w] = |N(u) ∩ N(w)|
        # After adding (u,v): N(u) = N_u ∪ {v}
        # So common[u,w]_new = |N_u ∩ N(w)| + (1 if v ∈ N(w) else 0)
        # v ∈ N(w) iff adj[w,v]=1 iff adj[v,w]=1
        
        # So for each w != u: if adj[v,w]=1, then common[u,w] increases by 1
        # Similarly for each w != v: if adj[u,w]=1, then common[v,w] increases by 1
        
        # We need to check if any pair that increases would reach 3
        
        # Compute current common counts for affected pairs
        # For efficiency, let's compute common matrix once
        common = adj @ adj
        
        # Check pairs (u, w) for w where adj[v,w]=1 and w != u
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                if common[u, w] >= 2:  # would become 3
                    return False
            if adj[u, w]:
                if common[v, w] >= 2:  # would become 3
                    return False
        
        # Also check pair (u,v) itself - but as argued, common[u,v] doesn't change
        # Actually wait: common[u,v] = |N(u) ∩ N(v)|. Before: N(u)=N_u, N(v)=N_v.
        # After: N(u)=N_u∪{v}, N(v)=N_v∪{u}.
        # N(u)∩N(v) = (N_u∪{v}) ∩ (N_v∪{u}) = (N_u∩N_v) ∪ (N_u∩{u}) ∪ ({v}∩N_v) ∪ ({v}∩{u})
        # N_u∩{u} = {} since u not in N_u
        # {v}∩N_v = {} since v not in N_v  
        # {v}∩{u} = {} since u≠v
        # So common[u,v] unchanged. Good.
        
        return True
    
    def add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
    
    def remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
    
    def edge_count(adj):
        return int(np.sum(adj)) // 2
    
    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    def greedy_add(adj):
        """Try to add all possible edges greedily."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            improved = True
        return adj
    
    def local_search(adj, max_iters=500):
        """Try edge swaps to improve."""
        best_adj = adj.copy()
        best_count = edge_count(best_adj)
        iters = 0
        while iters < max_iters and elapsed() < deadline - 5:
            iters += 1
            # Try removing one edge and adding another
            # Or try swapping: remove (u,v), add (a,b)
            # Simpler: try adding an edge by first removing a conflicting edge
            # Or: try all single edge additions (should already be maximal from greedy)
            # Try edge swaps: remove one edge, see if we can add two
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        # Try removing this edge
                        remove_edge(adj, u, v)
                        # Now try to add edges
                        added = 0
                        for a in range(n):
                            for b in range(a + 1, n):
                                if not adj[a, b]:
                                    if can_add_edge(adj, a, b):
                                        add_edge(adj, a, b)
                                        added += 1
                        if added > 1:
                            new_count = edge_count(adj)
                            if new_count > best_count:
                                best_adj = adj.copy()
                                best_count = new_count
                                found = True
                            else:
                                # Revert
                                for a in range(n):
                                    for b in range(a + 1, n):
                                        if adj[a, b] and not best_adj[a, b]:
                                            remove_edge(adj, a, b)
                                add_edge(adj, u, v)
                        else:
                            # Revert
                            add_edge(adj, u, v)
                if found:
                    break
                if elapsed() > deadline - 5:
                    break
            if not found:
                break
        return best_adj
    
    def palay_init():
        """Create a Paley-like graph. For n=20, use quadratic residues mod 19."""
        adj = np.zeros((n, n), dtype=int)
        # Use vertices 0..18 for Paley on 19 vertices (prime), vertex 19 separate
        p = 19
        # Quadratic residues mod 19
        residues = set()
        for i in range(1, p):
            residues.add((i * i) % p)
        for u in range(p):
            for v in range(p):
                if u != v and ((v - u) % p) in residues:
                    adj[u, v] = 1
        # Vertex 19: connect to some vertices
        # Try to connect vertex 19 to as many as possible
        for v in range(p):
            if can_add_edge(adj, 19, v):
                add_edge(adj, 19, v)
        return adj
    
    def random_init(density=0.5):
        """Random graph initialization."""
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    add_edge(adj, u, v)
        return adj
    
    def repair(adj):
        """Remove edges to fix K2,3 violations."""
        improved = True
        while improved:
            improved = False
            common = adj @ adj
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] >= 3:
                        # Need to remove an edge to reduce common neighbors
                        # Find a common neighbor w and remove (u,w) or (v,w)
                        common_neighbors = []
                        for w in range(n):
                            if w != u and w != v and adj[u, w] and adj[v, w]:
                                common_neighbors.append(w)
                        if common_neighbors:
                            # Remove edge to one of them
                            w = common_neighbors[0]
                            # Decide which edge to remove: remove the one that causes fewer future issues
                            # Simple: remove (u, w)
                            remove_edge(adj, u, w)
                            improved = True
                            break
                if improved:
                    break
        return adj
    
    def full_pipeline(adj):
        """Repair, greedy add, local search."""
        adj = repair(adj)
        adj = greedy_add(adj)
        adj = local_search(adj)
        return adj
    
    # Multiple strategies with restarts
    best_edges = []
    best_count = 0
    
    # Strategy 1: Paley initialization
    if elapsed() < deadline - 10:
        adj = palay_init()
        adj = full_pipeline(adj)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges_from_adj(adj)
    
    # Strategy 2: Multiple random restarts
    for restart in range(30):
        if elapsed() > deadline - 10:
            break
        density = random.uniform(0.3, 0.7)
        adj = random_init(density)
        adj = full_pipeline(adj)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges_from_adj(adj)
    
    # Strategy 3: Start from complete graph, remove edges to fix violations
    if elapsed() < deadline - 10:
        adj = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, 0)
        adj = repair(adj)
        adj = greedy_add(adj)
        adj = local_search(adj)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges_from_adj(adj)
    
    # Final verification
    if best_edges:
        adj_final = np.zeros((n, n), dtype=int)
        for e in best_edges:
            adj_final[e[0], e[1]] = 1
            adj_final[e[1], e[0]] = 1
        if not is_k23_free(adj_final):
            # Fallback to a safe construction
            adj_fallback = np.zeros((n, n), dtype=int)
            for k in range(5):
                for a in range(4):
                    for b in range(a + 1, 4):
                        adj_fallback[4*k+a, 4*k+b] = 1
                        adj_fallback[4*k+b, 4*k+a] = 1
            best_edges = edges_from_adj(adj_fallback)
    
    return best_edges
# EVOLVE-BLOCK-END

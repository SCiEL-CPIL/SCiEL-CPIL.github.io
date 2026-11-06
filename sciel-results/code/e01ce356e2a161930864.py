"""Construct C6-free graph on 40 vertices with maximum edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    n = 40
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    n_pairs = len(all_pairs)
    
    # Use bitmasks for fast adjacency checks
    # adj[i] is an integer where bit j is set if edge (i,j) exists
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def has_c6(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency adj creates a C6.
        A C6 through (u,v) has the form: u - a - b - c - d - v - u
        where a,b,c,d are distinct and not u or v.
        We need to check if there exist a in N(u)\{v}, d in N(v)\{u}, 
        and a path a-b-c-d of length 3 in G\{u,v}.
        """
        if (adj[u] >> v) & 1:
            return True  # Already an edge
        
        # Get neighbors of u and v excluding each other
        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)
        
        # For each a in N(u) and d in N(v), check for path a-b-c-d
        # b is neighbor of a, c is neighbor of d, b-c is edge
        # b,c not in {u,v,a,d}
        
        a_list = [a for a in range(n) if a != v and (nu_mask >> a) & 1]
        d_list = [d for d in range(n) if d != u and (nv_mask >> d) & 1]
        
        for a in a_list:
            for d in d_list:
                if a == d:
                    continue
                # Find b: neighbor of a, not in {u,v,a,d}
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                # Find c: neighbor of d, not in {u,v,a,d}
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                
                # Check if there exists b in b_mask and c in c_mask such that b-c is edge
                # b_mask & c_mask gives candidates where b=c, but we need b-c edge
                # Iterate over b candidates
                b_list = [b for b in range(n) if (b_mask >> b) & 1]
                for b in b_list:
                    # Check if b is connected to any c in c_mask
                    bc_mask = adj[b] & c_mask
                    if bc_mask:
                        return True
        return False
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if not has_c6(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        """Add edge (u,v) without checking."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def get_edges(adj):
        """Get list of edges from adjacency masks."""
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def local_search(adj, max_iter=2000):
        """
        Local search: try to add any missing edge that doesn't create C6.
        """
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if time.time() > deadline:
                    return adj
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
        return adj
    
    random.seed(42)
    best_adj = None
    best_count = 0
    
    # Phase 1: Multiple random restarts with greedy construction
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        # Greedy add
        for u, v in shuffled:
            if time.time() > deadline:
                break
            try_add_edge(adj, u, v)
        
        # Local search to fill in missing edges
        adj = local_search(adj)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        if best_count >= 105:
            # Try a few more restarts to see if we can do better
            if restart >= 10:
                break
    
    # Phase 2: Perturbation-based local search from best solution
    if best_adj is not None:
        for perturb_round in range(30):
            if time.time() > deadline:
                break
            
            adj = best_adj[:]
            
            # Remove k random edges to perturb
            k = random.randint(3, 8)
            current_edges = get_edges(adj)
            if len(current_edges) > k:
                removed = random.sample(current_edges, k)
                for u, v in removed:
                    remove_edge(adj, u, v)
            
            # Try to re-add edges greedily
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    try_add_edge(adj, u, v)
            
            # Local search
            adj = local_search(adj)
            
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
    
    return get_edges(best_adj)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")

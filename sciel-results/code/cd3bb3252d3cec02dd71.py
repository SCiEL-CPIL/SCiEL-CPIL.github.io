import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0] * n  # bitmasks for adjacency
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_clique_partition(adj, n):
        # adj is a list of bitmasks
        # remaining_edges is a set of (u, v) tuples? 
        # Using bitmasks is faster for checking, but we need to track which edges are removed.
        # Let's keep a copy of adj and clear bits as we remove edges.
        # Actually, we just need to find a clique in the *current* graph.
        # The current graph is defined by the remaining edges.
        # We can maintain `cur_adj` which is a list of bitmasks representing the current graph.
        
        cur_adj = list(adj)
        cliques = []
        
        while True:
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if cur_adj[u] != 0:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find a maximal clique in the current graph
            # Heuristic: Start with a vertex of max degree, then greedily add neighbors
            # 1. Find vertex with max degree in cur_adj
            max_deg = -1
            start_v = -1
            for u in range(n):
                deg = bin(cur_adj[u]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            
            if max_deg == 0:
                # This shouldn't happen if has_edges is True, but just in case
                break
                
            # Initialize clique candidates
            # The clique must be a subset of {start_v} U N(start_v)
            # Let's use a greedy set construction
            # Candidates: start_v and its neighbors
            # We want to pick a set of vertices that are mutually adjacent.
            
            # A simple greedy:
            # C = {start_v}
            # Candidates = N(start_v)
            # While Candidates is not empty:
            #    Pick v in Candidates with max degree within Candidates (or global?)
            #    Add v to C
            #    Remove v and non-neighbors of v from Candidates
            
            C = [start_v]
            # Candidates are neighbors of start_v that are still in the graph
            # Note: cur_adj[start_v] contains neighbors.
            candidates = cur_adj[start_v]
            
            while candidates:
                # Find vertex in candidates with max degree within candidates
                # Degree within candidates = popcount(cur_adj[v] & candidates)
                best_v = -1
                best_deg = -1
                c = candidates
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    deg_in_cand = bin(cur_adj[v] & candidates).count('1')
                    if deg_in_cand > best_deg:
                        best_deg = deg_in_cand
                        best_v = v
                    c ^= lsb
                
                if best_v == -1:
                    break
                    
                C.append(best_v)
                
                # Update candidates: must be neighbors of best_v and still in candidates
                # Also, we removed best_v from candidates
                candidates &= cur_adj[best_v]
                candidates &= ~(1 << best_v)
                
            # C is now a clique
            # Verify C is a clique (safety check, should always be true)
            # Remove edges of C from cur_adj
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    u, v = C[i], C[j]
                    cur_adj[u] &= ~(1 << v)
                    cur_adj[v] &= ~(1 << u)
            
            cliques.append(C)
            
            # Time check
            if time.time() > deadline:
                break
                
        return cliques

    # Instance 1
    n1, p1, seed1 = 32, 0.6, 601
    adj1 = random_graph(n1, p1, seed1)
    cliques1 = greedy_clique_partition(adj1, n1)
    
    # Instance 2
    n2, p2, seed2 = 64, 0.55, 602
    adj2 = random_graph(n2, p2, seed2)
    cliques2 = greedy_clique_partition(adj2, n2)
    
    return [cliques1, cliques2]

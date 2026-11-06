import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0]*n
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def greedy_large_clique(adj, mask):
        """
        Finds a large clique in the subgraph induced by vertices in mask.
        Uses a greedy heuristic: repeatedly pick the vertex with highest degree 
        within the current candidate set that is connected to all selected vertices.
        """
        if not mask:
            return []
        
        # Get list of vertices in mask
        candidates = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        clique = []
        # We need to maintain the set of candidates that are connected to ALL vertices in clique
        # Initially, all candidates are valid.
        # As we add v to clique, new_candidates = candidates & adj[v]
        
        # To make it "large", we pick the one with highest degree in the current valid set.
        # We will simulate the Bron-Kerbosch idea but greedily.
        
        current_candidates = set(candidates)
        
        while current_candidates:
            # Pick vertex v in current_candidates with max degree within current_candidates
            # Note: Degree within current_candidates means number of neighbors in current_candidates
            best_v = None
            max_deg = -1
            for v in current_candidates:
                # Count neighbors in current_candidates
                # adj[v] & mask_of_current_candidates
                # We can just iterate since n is small (32/64)
                deg = 0
                for u in current_candidates:
                    if u == v: continue
                    if adj[v] & (1 << u):
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    
            if best_v is None:
                break
                
            clique.append(best_v)
            current_candidates.remove(best_v)
            
            # Update candidates: must be neighbor of best_v
            new_candidates = set()
            for u in current_candidates:
                if adj[best_v] & (1 << u):
                    new_candidates.add(u)
            current_candidates = new_candidates
            
        return clique

    def solve_instance(n, p, seed, time_limit):
        adj, edges = random_graph(n, p, seed)
        
        # rem_adj[u] is a set of neighbors v such that (u,v) is still in the graph
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        remaining_edges_count = len(edges)
        
        best_partition = None
        best_score = float('inf')
        
        # Run multiple restarts with different random seeds for tie-breaking
        # The base seed determines the graph. We use a different seed for the heuristic choices.
        # Actually, the graph is fixed. We just vary the heuristic.
        
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Copy state
            curr_rem_adj = [set(s) for s in rem_adj]
            curr_edges = remaining_edges_count
            partition = []
            
            while curr_edges > 0:
                if time.time() > deadline - 5.0:
                    break
                    
                # Find vertex with max degree
                # Add some randomness to tie-breaking
                max_deg = max(len(s) for s in curr_rem_adj)
                if max_deg == 0:
                    break
                
                candidates = [i for i in range(n) if len(curr_rem_adj[i]) == max_deg]
                # Randomize selection among max degree vertices
                random.shuffle(candidates)
                v = candidates[0]
                
                neighbors = list(curr_rem_adj[v])
                if not neighbors:
                    continue
                    
                # Build mask for neighbors
                mask = 0
                for u in neighbors:
                    mask |= (1 << u)
                    
                # Find large clique in neighbors
                clique_in_neighbors = greedy_large_clique(adj, mask)
                
                if not clique_in_neighbors:
                    # Should not happen if neighbors is not empty and graph is simple, 
                    # unless no edges? But neighbors are connected to v.
                    # If clique_in_neighbors is empty, we can't form a clique with v.
                    # But v is connected to all in neighbors. So {v} is a clique of size 1?
                    # Task says "clique with at least two vertices".
                    # So we must form a clique of size >= 2.
                    # If greedy fails (returns empty), pick any neighbor u.
                    # {v, u} is a clique.
                    u = neighbors[0]
                    clique = [v, u]
                else:
                    clique = [v] + clique_in_neighbors
                    
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, w = clique[i], clique[j]
                        if w in curr_rem_adj[u]:
                            curr_rem_adj[u].remove(w)
                            curr_rem_adj[w].remove(u)
                            curr_edges -= 1
                            
                partition.append(clique)
                
            # Fallback for any remaining edges (should be few if loop completed)
            for u in range(n):
                for v in list(curr_rem_adj[u]):
                    if u < v:
                        partition.append([u, v])
                        curr_rem_adj[u].remove(v)
                        curr_rem_adj[v].remove(u)
                        
            score = len(partition)
            if score < best_score:
                best_score = score
                best_partition = partition
                
        # If no partition found (e.g. timeout), return a valid but potentially suboptimal one
        if best_partition is None:
            # Just do a single greedy pass quickly
            curr_rem_adj = [set(s) for s in rem_adj]
            curr_edges = remaining_edges_count
            partition = []
            while curr_edges > 0:
                max_deg = max(len(s) for s in curr_rem_adj)
                if max_deg == 0: break
                v = next(i for i in range(n) if len(curr_rem_adj[i]) == max_deg)
                neighbors = list(curr_rem_adj[v])
                if not neighbors: continue
                mask = 0
                for u in neighbors: mask |= (1 << u)
                c = greedy_large_clique(adj, mask)
                clique = [v] + (c if c else [neighbors[0]])
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, w = clique[i], clique[j]
                        if w in curr_rem_adj[u]:
                            curr_rem_adj[u].remove(w)
                            curr_rem_adj[w].remove(u)
                            curr_edges -= 1
                partition.append(clique)
            for u in range(n):
                for v in list(curr_rem_adj[u]):
                    if u < v:
                        partition.append([u, v])
                        curr_rem_adj[u].remove(v)
                        curr_rem_adj[v].remove(u)
            best_partition = partition
            
        return best_partition

    result1 = solve_instance(32, 0.6, 601, deadline)
    result2 = solve_instance(64, 0.55, 602, deadline)
    
    return [result1, result2]

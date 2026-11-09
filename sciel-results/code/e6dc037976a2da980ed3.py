import math
import random
import time

def construct():
    N = 255
    # Candidates must share a factor with 255
    # Primes of 255: 3, 5, 17
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    
    # We must include 255.
    # The problem is to find the max clique in the graph induced by candidates, 
    # such that 255 is in it. Since 255 is connected to all candidates,
    # this is equivalent to finding the max clique in the graph induced by candidates.
    # Wait, if the max clique in candidates is K, then K U {255} is a valid set.
    # So we just need max clique in G[candidates].
    
    # However, we can just search for max clique in the whole graph [1..255] 
    # constrained to include 255.
    
    # Let's use a heuristic:
    # 1. Try the set of all multiples of 3. Size 85.
    # 2. Try the set of all multiples of 5. Size 51.
    # 3. Try the set of all multiples of 17. Size 15.
    # 4. Try a greedy clique construction.
    
    best_set = [x for x in range(1, N + 1) if x % 3 == 0]
    best_size = len(best_set)
    
    # Greedy Max Clique Heuristic
    # Graph vertices: candidates
    # Edge: gcd(u,v) > 1
    
    # Precompute gcd matrix for candidates?
    # Candidates are ~127 nodes. 127x127 matrix is small.
    
    n_cand = len(candidates)
    # Map candidate index to value
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    def greedy_clique(start_node=None, random_order=False):
        # Returns a clique (list of indices)
        if random_order:
            nodes = list(range(n_cand))
            random.shuffle(nodes)
        else:
            # Sort by degree descending
            nodes = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
            
        if start_node is not None:
            # Ensure start_node is first
            nodes.remove(start_node)
            nodes.insert(0, start_node)
            
        clique = []
        if start_node is not None:
            clique.append(start_node)
            candidates_for_next = set(adj[start_node])
        else:
            candidates_for_next = set(nodes) # All nodes are candidates initially
            
        for node in nodes:
            if node in candidates_for_next:
                clique.append(node)
                # Update candidates_for_next to intersection with adj[node]
                candidates_for_next &= adj[node]
                
        return clique

    # We need to ensure 255 is in the final set.
    # 255 is in candidates. Let's find its index.
    idx_255 = candidates.index(255)
    
    # Try greedy starting with 255
    clique_indices = greedy_clique(start_node=idx_255)
    clique_vals = [candidates[i] for i in clique_indices]
    if 255 not in clique_vals:
        clique_vals.append(255) # Should not happen if start_node=idx_255
    size = len(clique_vals)
    if size > best_size:
        best_set = clique_vals
        best_size = size

    # Try random restarts
    start_time = time.time()
    while time.time() - start_time < 50: # 50s budget
        clique_indices = greedy_clique(start_node=idx_255, random_order=True)
        clique_vals = [candidates[i] for i in clique_indices]
        if 255 not in clique_vals:
            clique_vals.append(255)
        size = len(clique_vals)
        if size > best_size:
            best_set = clique_vals
            best_size = size
            # Optional: local search improvement here?
            
    return best_set

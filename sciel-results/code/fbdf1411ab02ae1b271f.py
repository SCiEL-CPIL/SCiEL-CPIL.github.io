import time
import random
from collections import defaultdict
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe against 60s kill
    
    n = 12
    
    def get_cycle_lengths_fast(edges):
        """
        Returns a set of cycle lengths. 
        Optimized for small graphs (n=12).
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # To avoid counting the same cycle multiple times, we only start DFS 
        # from the smallest vertex in the cycle.
        # A cycle is identified by its vertices. 
        # We can find all simple cycles using Johnson's algorithm or a simple DFS 
        # with pruning for n=12.
        
        # Given n=12, a simple DFS is fine.
        # We restrict to cycles where the starting node is the minimum node in the cycle.
        
        for start in range(n):
            # DFS from start, only visiting nodes > start, 
            # except when closing the cycle back to start.
            # This ensures each cycle is found exactly once (rooted at min node).
            
            def dfs(node, path, visited):
                # If we can close the cycle
                # The cycle length is len(path)
                # We only care about distinct lengths.
                
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 4: # Girth >= 4
                        cycle_lengths.add(len(path))
                    elif neighbor > start and neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            
            dfs(start, [start], {start})
            
        return cycle_lengths

    def is_valid_fast(edges):
        """
        Checks:
        1. Simple (no loops, no multi-edges)
        2. Degree >= 3
        3. Triangle-free (Girth >= 4)
        4. Has a cycle
        """
        n = 12
        edge_set = set()
        deg = [0] * n
        
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            
        for d in deg:
            if d < 3:
                return False
        
        # Check triangle-free
        # For each edge (u,v), check if they have a common neighbor.
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        for u, v in edge_set:
            # Intersection of neighbors
            if adj[u] & adj[v]:
                return False
                
        # Check has cycle
        # In a 3-regular graph on 12 vertices, if it's valid, it likely has a cycle.
        # But strictly, a tree has no cycles. Max edges in tree is 11.
        # We have 18 edges (3*12/2). So it MUST have a cycle.
        # So this check is redundant for 3-regular graphs, but good for safety.
        return True

    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges

    # Default best: Two K3,3 components. Cycle lengths {4, 6}. Raw value 2.
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2
    
    # Search for a graph with 1 distinct cycle length.
    # We search among 3-regular bipartite graphs on 12 vertices.
    # Parts: A={0..5}, B={6..11}.
    
    random.seed(42)
    
    # Strategy: 
    # 1. Try random 3-regular bipartite graphs.
    #    Represented by a 6x6 matrix M where M[i][j]=1 if i connected to 6+j.
    #    Row sums 3, Col sums 3.
    
    # To generate random 3-regular bipartite graphs:
    # We can use the configuration model or random permutations.
    # A simple way: 
    # Start with a valid one (e.g., K3,3 is not 3-regular on 6+6? Yes it is).
    # Actually, K3,3 is 3-regular.
    # We can apply random row/column permutations? No, that's isomorphism.
    # We need to change the structure.
    
    # Let's try generating random 3-regular bipartite graphs by:
    # 1. Creating a random 3-regular graph on 12 vertices and checking if it's bipartite.
    # 2. Or, more efficiently, generating random 0-1 matrices with row/col sum 3.
    
    def generate_random_3reg_bipartite():
        # Use a simple rejection sampling or a constructive method.
        # Constructive:
        # Let A be 0..5, B be 6..11.
        # We need to assign 3 neighbors from B to each i in A.
        # And each j in B must have 3 neighbors from A.
        
        # This is equivalent to finding a random 3-regular bipartite graph.
        # We can model this as a random permutation of a multiset?
        # Or use the "swap" method:
        # Start with a valid graph (e.g., three disjoint perfect matchings).
        # M = I + P + P^2 for some permutation P?
        
        # Let's try: 
        # Base: 3 perfect matchings.
        # M1: i -> 6+i
        # M2: i -> 6+(i+1)%6
        # M3: i -> 6+(i+2)%6
        # This is a valid 3-regular bipartite graph.
        # We can randomize the permutations for M2 and M3?
        
        # Actually, just generate random permutations p1, p2, p3 such that 
        # for each i, {p1(i), p2(i), p3(i)} are distinct, and for each j, 
        # the number of i such that j in {p1(i), p2(i), p3(i)} is 3.
        
        # This is complex to enforce directly.
        # Let's stick to the permutation shift method used in the current program, 
        # but expand the search space significantly.
        
        perm = list(range(6))
        random.shuffle(perm)
        
        edges = []
        valid = True
        seen = set()
        
        for i in range(6):
            # Neighbors are perm[(i+k) % 6] for k in {0,1,2}
            # This ensures degree 3 for A.
            # Does it ensure degree 3 for B?
            # Yes, because the mapping i -> perm[(i+k)%6] is a bijection for each k.
            # So each B vertex gets exactly one edge from each k-shift. Total 3.
            
            for k in range(3):
                u = i
                v = 6 + perm[(i + k) % 6]
                if u == v: # Should not happen as u<6, v>=6
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append([u, v])
            if not valid:
                break
        
        if not valid:
            return None
            
        return edges

    # Search loop
    # We will try many random graphs.
    # If we find one with 1 cycle length, we return it immediately.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        edges = generate_random_3reg_bipartite()
        if edges is None:
            continue
            
        # Check validity
        if not is_valid_fast(edges):
            continue
            
        # Check cycle lengths
        cl = get_cycle_lengths_fast(edges)
        
        if not cl:
            continue
            
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edges
            if score == 1:
                break

    # Fallback is already set to K3,3 + K3,3 (score 2)
    
    # Ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
# EVOLVE-BLOCK-END

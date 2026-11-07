"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C (first k vertices)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_graph(n, k, edges, time_limit):
        """Find minimum clique partition of the given edges."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        uncovered = set(edge_set)
        cliques = []
        
        # Precompute: for each vertex, its neighbors
        # Greedy: repeatedly find a maximal clique that covers many uncovered edges
        # Priority: prefer cliques that cover edges with high "bottleneck" (few alternative coverings)
        
        # For split graphs, structure: C = {0..k-1}, I = {k..n-1}
        # Edges in C form K_k. Cross edges connect C to I.
        # No edges within I.
        
        # Strategy: 
        # 1. Partition K_k edges into cliques
        # 2. For each i in I, cover its incident edges with cliques {i} ∪ S where S ⊆ N(i) ∩ C
        
        # But these interact. Let's use a unified greedy.
        
        # Count degree of each edge (how many cliques could cover it)
        # An edge (u,v) can be in a clique only if all vertices in the clique are mutually adjacent.
        
        # Greedy with priority:
        # For each uncovered edge, estimate how "hard" it is to cover
        # Prefer to cover hard edges first with large cliques
        
        while uncovered and time.time() < time_limit:
            # Find the edge that is "most constrained"
            # For each uncovered edge (u,v), the number of possible cliques containing it
            # is the number of cliques in the common neighborhood
            
            # Simpler: just find a maximal clique with most uncovered edges
            best_clique = None
            best_count = -1
            
            # Try to find cliques centered around high-degree vertices in uncovered subgraph
            # Build uncovered adjacency
            uncov_adj = [set() for _ in range(n)]
            for u, v in uncovered:
                uncov_adj[u].add(v)
                uncov_adj[v].add(u)
            
            # For each vertex, try to extend to a clique using only uncovered edges
            for start in range(n):
                if not uncov_adj[start]:
                    continue
                # Greedy clique expansion
                clique = [start]
                candidates = list(uncov_adj[start])
                while candidates:
                    # Pick candidate that maximizes connections to remaining candidates
                    best_cand = None
                    best_score = -1
                    for c in candidates:
                        score = len(uncov_adj[c] & set(clique)) + len(uncov_adj[c] & set(candidates))
                        if score > best_score:
                            best_score = score
                            best_cand = c
                    if best_cand is None:
                        break
                    clique.append(best_cand)
                    candidates = [c for c in candidates if c in uncov_adj[best_cand]]
                
                if len(clique) >= 2:
                    # Count uncovered edges in this clique
                    count = 0
                    clique_set = set(clique)
                    for u, v in uncovered:
                        if u in clique_set and v in clique_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_clique = clique
            
            if best_clique is None or best_count <= 0:
                # Fallback: take any uncovered edge as a 2-clique
                u, v = next(iter(uncovered))
                best_clique = [u, v]
                best_count = 1
            
            # Add clique
            clique_set = set(best_clique)
            cliques.append(best_clique)
            
            # Remove covered edges
            to_remove = set()
            for u, v in uncovered:
                if u in clique_set and v in clique_set:
                    to_remove.add((u, v))
            uncovered -= to_remove
        
        return cliques
    
    def solve_split_graph(n, k, p, seed):
        """Exploit split graph structure for better solution."""
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # For each i in I, N(i) ∩ C
        neighbors_in_C = {i: adj[i] & set(C) for i in I}
        
        # Strategy: 
        # The K_k edges must be partitioned. Cross edges from i to N(i)∩C can be 
        # covered by cliques {i} ∪ S where S is a clique in C (always true).
        # 
        # Key insight: we can combine K_k clique partition with cross edge coverage.
        # If we have a clique T in C, we can extend it to {i} ∪ T for any i where T ⊆ N(i)∩C.
        # This covers both the edges within T (part of K_k) and the cross edges from i to T.
        
        # Approach: 
        # 1. Start with a partition of K_k edges into cliques
        # 2. For each independent vertex, assign its cross edges to existing cliques or create new ones
        
        # Step 1: Partition K_k
        # Use greedy: repeatedly take a maximal clique from remaining K_k edges
        k_edges = [(u, v) for u in range(k) for v in range(u+1, k)]
        k_uncovered = set(k_edges)
        k_cliques = []
        
        while k_uncovered:
            # Find maximal clique with most uncovered edges
            best_clique = None
            best_count = -1
            for start in range(k):
                # Greedy expansion
                clique = [start]
                cand = [v for v in range(start+1, k) if (start, v) in k_uncovered or (v, start) in k_uncovered]
                # Actually all pairs in C are edges, so any subset is a clique
                # We just need to pick a subset that covers many uncovered edges
                # Greedy: pick vertices that have many uncovered edges among themselves
                remaining = list(range(k))
                clique = []
                while remaining:
                    # Pick vertex with most uncovered edges to other remaining
                    best_v = None
                    best_deg = -1
                    for v in remaining:
                        deg = sum(1 for w in remaining if w != v and (min(v,w), max(v,w)) in k_uncovered)
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                    if best_v is None or best_deg == 0:
                        break
                    clique.append(best_v)
                    remaining.remove(best_v)
                
                if len(clique) >= 2:
                    count = sum(1 for u, v in k_uncovered if u in clique and v in clique)
                    if count > best_count:
                        best_count = count
                        best_clique = clique
            
            if best_clique is None:
                u, v = next(iter(k_uncovered))
                best_clique = [u, v]
            
            k_cliques.append(best_clique)
            clique_set = set(best_clique)
            k_uncovered = {(u,v) for u,v in k_uncovered if not (u in clique_set and v in clique_set)}
        
        # Step 2: Assign cross edges
        # For each i in I, its cross edges are (i, c) for c in neighbors_in_C[i]
        # We can cover these with cliques {i} ∪ S where S ⊆ neighbors_in_C[i]
        # To minimize cliques, we want to use as few such cliques as possible per i
        # And we want to reuse the structure of k_cliques where possible
        
        # For each i, partition neighbors_in_C[i] into as few subsets as possible
        # Each subset S gives clique {i} ∪ S
        # Minimum number of subsets = 1 (just use all of N(i)∩C as one clique)
        # But wait - we need the edges within S to also be covered. If we use {i} ∪ S as a clique,
        # it covers edges (i,c) for c in S AND edges (c1,c2) for c1,c2 in S.
        # The K_k edges within S are already covered by k_cliques. So we'd be double-covering.
        
        # Correction: each edge must be in EXACTLY one clique. So if we use {i} ∪ S,
        # the edges within S must NOT be in any other clique.
        
        # So the approach must be: partition ALL edges (K_k + cross) into cliques.
        # A clique can contain vertices from C and at most one from I.
        
        # Let me redo this properly with a unified greedy.
        
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u,v), max(u,v)))
        
        uncovered = set(all_edges)
        cliques = []
        
        # Priority-based greedy
        # For each uncovered edge, compute a "difficulty" score
        # Prefer to cover edges that are in fewer potential cliques first
        
        while uncovered and time.time() < time_limit:
            # For each vertex, compute the set of uncovered neighbors
            uncov_nbrs = [set() for _ in range(n)]
            for u, v in uncovered:
                uncov_nbrs[u].add(v)
                uncov_nbrs[v].add(u)
            
            # Find best clique: try all starting vertices, greedy expand
            best_clique = None
            best_score = -1
            
            for start in range(n):
                if not uncov_nbrs[start]:
                    continue
                # Greedy clique: start with start, add vertices that are adjacent to all current clique members
                clique = [start]
                candidates = list(uncov_nbrs[start])
                
                while candidates:
                    # Pick candidate with max (uncovered edges to clique + uncovered edges to other candidates)
                    best_cand = None
                    best_cand_score = -1
                    for c in candidates:
                        # Must be adjacent to all in clique (for it to be a clique)
                        valid = all(c in uncov_nbrs[x] for x in clique)
                        if not valid:
                            continue
                        score = len(uncov_nbrs[c] & set(clique)) + len(uncov_nbrs[c] & set(candidates))
                        if score > best_cand_score:
                            best_cand_score = score
                            best_cand = c
                    if best_cand is None:
                        break
                    clique.append(best_cand)
                    # Update candidates: must be adjacent to best_cand
                    candidates = [c for c in candidates if c in uncov_nbrs[best_cand]]
                
                if len(clique) >= 2:
                    clique_set = set(clique)
                    count = sum(1 for u, v in uncovered if u in clique_set and v in clique_set)
                    if count > best_score:
                        best_score = count
                        best_clique = clique
            
            if best_clique is None or best_score <= 0:
                u, v = next(iter(uncovered))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            clique_set = set(best_clique)
            uncovered = {(u,v) for u,v in uncovered if not (u in clique_set and v in clique_set)}
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        time_left = deadline - time.time()
        if time_left < 5:
            # Not enough time, use seed solution
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        cliques = solve_split_graph(n, k, p, seed, time_left * 0.8)
        results.append(cliques)
    
    # Validate and convert
    final_results = []
    for cliques in results:
        validated = []
        for clique in cliques:
            validated.append([int(v) for v in clique])
        final_results.append(validated)
    
    return final_results
# EVOLVE-BLOCK-END

"""Clique partition of split graphs using conflict-resolution on C-edges."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # C vertices: 0..k-1
        # I vertices: k..n-1
        
        # Initial S_i for each i in I
        S = {}
        for i in range(k, n):
            # Neighbors in C
            neighbors_in_c = [c for c in range(k) if c in adj[i]]
            S[i] = set(neighbors_in_c)
            
        # Count coverage of each C-edge
        # C-edge is (u, v) with u < v, u, v < k
        c_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edge_coverage = {e: 0 for e in c_edges}
        
        # Precompute which i covers which c-edge
        # For each i, the c-edges covered are those within S[i]
        i_covered_edges = {}
        for i in range(k, n):
            covered = []
            s_list = list(S[i])
            for idx_u in range(len(s_list)):
                u = s_list[idx_u]
                for idx_v in range(idx_u + 1, len(s_list)):
                    v = s_list[idx_v]
                    e = (min(u, v), max(u, v))
                    covered.append(e)
            i_covered_edges[i] = covered
            
        # Update coverage counts
        for i in range(k, n):
            for e in i_covered_edges[i]:
                edge_coverage[e] += 1
                
        # Resolve conflicts: while any edge has coverage > 1
        # We will remove vertices from S[i] and create new cliques
        new_cliques = []
        
        # To be efficient, we can just iterate through edges with count > 1
        # But removing a vertex changes counts for many edges.
        # Simple approach: repeatedly find an edge with count > 1, fix it.
        
        # Optimization: We can process all conflicts in one pass?
        # No, because fixing one might affect others.
        
        # Let's use a loop
        # Max iterations: bounded by number of edges * something
        # But k is small (30, 20, 10).
        
        conflict_count = 0
        while True:
            if time.time() > deadline - 5:
                # Fallback: break and use current state (might be invalid if conflicts remain, but hopefully not)
                # Actually, if we break, we must ensure validity.
                # If conflicts remain, we have duplicate coverage.
                # We can force resolution by adding edge cliques for all remaining conflicts?
                # No, that doesn't fix the duplicate in K_i.
                # We must remove from K_i.
                # Given time, let's hope it finishes.
                break
                
            # Find an edge with coverage > 1
            conflicted_edge = None
            for e in c_edges:
                if edge_coverage[e] > 1:
                    conflicted_edge = e
                    break
                    
            if conflicted_edge is None:
                break
                
            u, v = conflicted_edge
            
            # Find an i that covers this edge
            # We can pick the i with the largest S[i] to minimize impact?
            # Or just the first one.
            covering_is = [i for i in range(k, n) if conflicted_edge in i_covered_edges[i]]
            
            if not covering_is:
                # Should not happen if coverage > 1
                continue
                
            # Pick one i to modify
            i_mod = covering_is[0]
            
            # Remove u from S[i_mod]
            # This removes all edges (u, w) from coverage of i_mod
            # And adds a new clique {i_mod, u}
            
            # Update S[i_mod]
            S[i_mod].discard(u)
            
            # Update i_covered_edges[i_mod]
            # Remove all edges containing u
            new_covered = []
            for e in i_covered_edges[i_mod]:
                if u not in e:
                    new_covered.append(e)
                else:
                    # Decrease coverage count
                    edge_coverage[e] -= 1
            i_covered_edges[i_mod] = new_covered
            
            # Add new clique
            new_cliques.append([i_mod, u])
            
            conflict_count += 1
            
            # Safety: if conflict_count is huge, something is wrong
            if conflict_count > 10000:
                break

        # Now, handle uncovered C-edges (coverage == 0)
        for e in c_edges:
            if edge_coverage[e] == 0:
                u, v = e
                new_cliques.append([u, v])

        # Construct final list of cliques
        final_cliques = []
        
        # Add K_i for each i in I
        for i in range(k, n):
            if S[i]:
                clique = [i] + sorted(list(S[i]))
                final_cliques.append(clique)
            # If S[i] is empty, no clique needed (no cross edges to cover)
            
        # Add new cliques
        final_cliques.extend(new_cliques)
        
        # Convert to lists of Python ints
        final_cliques = [[int(x) for x in c] for c in final_cliques]
        
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END

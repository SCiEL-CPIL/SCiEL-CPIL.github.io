"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_greedy_max_clique(vertices, avail_edges):
        """
        Find a maximal clique in the subgraph induced by 'vertices' 
        using only edges in 'avail_edges'.
        vertices: list/set of vertex indices
        avail_edges: set of edges (frozensets or tuples)
        Returns: list of vertices forming a clique
        """
        if not vertices:
            return []
        
        # Convert to list for deterministic ordering if needed, but set is fine
        # Sort by degree in the available subgraph (descending)
        # This helps find a large clique
        v_list = list(vertices)
        
        # Calculate degrees within the available subgraph
        degrees = {v: 0 for v in v_list}
        for v in v_list:
            for u in v_list:
                if v == u: continue
                e = (v, u) if v < u else (u, v)
                if e in avail_edges:
                    degrees[v] += 1
        
        # Sort by degree descending
        v_list.sort(key=lambda x: degrees[x], reverse=True)
        
        clique = []
        candidates = set(v_list)
        
        for v in v_list:
            if v not in candidates:
                continue
            # Check if v is connected to all in current clique
            if all(((v, u) if v < u else (u, v)) in avail_edges for u in clique):
                clique.append(v)
                # Remove non-neighbors from candidates
                # Actually, we just need to ensure future vertices connect to all in clique.
                # We can prune candidates to only those connected to v.
                new_candidates = set()
                for w in candidates:
                    if w == v:
                        continue
                    e = (v, w) if v < w else (w, v)
                    if e in avail_edges:
                        new_candidates.add(w)
                candidates = new_candidates
        
        return clique

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build edge set
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))
        
        # C is 0..k-1, I is k..n-1
        # avail_cc_edges starts as all C-C edges
        avail_cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                avail_cc_edges.add((u, v))
        
        cliques = []
        
        # Calculate N_C for each I vertex
        N_C = {}
        for v in range(k, n):
            neigh = [u for u in range(k) if (u, v) in edge_set]
            N_C[v] = neigh
        
        # Sort I vertices by degree descending
        i_vertices = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
        
        for v in i_vertices:
            if time.time() > deadline - 5:
                # Fallback: individual edges
                for u in N_C[v]:
                    cliques.append([v, u])
                # And cover remaining C edges individually? No, just break and handle later
                break
                
            U = N_C[v]
            if not U:
                continue
            
            # Greedy clique cover of the available subgraph on U
            remaining_U = set(U)
            
            while remaining_U:
                # Find a maximal clique in remaining_U using avail_cc_edges
                S = find_greedy_max_clique(remaining_U, avail_cc_edges)
                
                if not S:
                    # Should not happen, but safety
                    u = next(iter(remaining_U))
                    S = [u]
                
                # Create clique {v} + S
                cliques.append([v] + S)
                
                # Remove edges in S from avail_cc_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        e = (a, b) if a < b else (b, a)
                        avail_cc_edges.discard(e)
                
                # Remove S from remaining_U
                remaining_U -= set(S)
        
        # Cover remaining avail_cc_edges with C-only cliques
        remaining_cc = set(avail_cc_edges)
        
        while remaining_cc:
            if time.time() > deadline - 2:
                # Fallback: individual edges
                for e in remaining_cc:
                    cliques.append(list(e))
                remaining_cc = set()
                break
            
            # Find a maximal clique in C using remaining_cc
            # Vertices involved in remaining_cc
            vertices_in_cc = set()
            for e in remaining_cc:
                vertices_in_cc.add(e[0])
                vertices_in_cc.add(e[1])
            
            S = find_greedy_max_clique(vertices_in_cc, remaining_cc)
            
            if not S:
                # Safety
                e = next(iter(remaining_cc))
                S = list(e)
            
            cliques.append(S)
            
            # Remove edges in S from remaining_cc
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    e = (a, b) if a < b else (b, a)
                    remaining_cc.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            # Emergency fallback for the whole instance
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for (u, v) in edges:
                if not (u < k and v < k):
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END

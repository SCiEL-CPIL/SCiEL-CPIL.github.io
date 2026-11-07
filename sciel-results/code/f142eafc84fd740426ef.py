"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
            
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # 1. Parse Graph Structure
        # C = {0, ..., k-1}, I = {k, ..., n-1}
        # Edges are sorted.
        # We need to identify E_CC (edges in C) and E_CI (edges between C and I).
        
        E_CC = set()
        E_CI = {} # v in I -> set of neighbors in C
        
        for u, v in edges:
            if u < k and v < k:
                E_CC.add((u, v))
            elif u < k and v >= k:
                if v not in E_CI:
                    E_CI[v] = set()
                E_CI[v].add(u)
            elif v < k and u >= k:
                if u not in E_CI:
                    E_CI[u] = set()
                E_CI[u].add(v)
        
        # 2. Strategy:
        # We want to partition E_CC ∪ E_CI into cliques.
        # A clique K can be of two types:
        # Type A: Subset of C only. Covers only edges in C.
        # Type B: {v} ∪ S, where v in I and S subset of N(v) in C.
        #         Covers edges (u, v) for u in S, and edges (u1, u2) for u1, u2 in S.
        
        # Key Insight:
        # If we use Type B cliques, they "consume" edges in C.
        # Specifically, if K = {v} ∪ S, it consumes all edges in the complete graph on S.
        # The remaining edges in C must be covered by Type A cliques (or other Type B cliques that don't conflict, but since I vertices are distinct, Type B cliques for different v's only conflict on C edges if they share >=2 vertices in C).
        
        # To minimize total cliques:
        # 1. Maximize the "efficiency" of Type B cliques.
        #    A Type B clique with |S| = m covers m edges in CI and m(m-1)/2 edges in CC.
        #    Cost: 1 clique.
        #    Savings vs size-2:
        #    - For CI edges: saves m-1 cliques.
        #    - For CC edges: saves m(m-1)/2 - 1 cliques (if we had to cover them with size-2s).
        #    So larger m is better.
        
        # 2. However, Type B cliques for different v's cannot share edges in C.
        #    This means if K_v uses S_v, then K_w uses S_w, we must have |S_v ∩ S_w| <= 1.
        #    (If they share 2 vertices, the edge between them is covered twice).
        
        # Algorithm:
        # Step A: Determine a set of disjoint (in terms of C-edges) subsets S_v for each v in I.
        #         Ideally, we want S_v to be large.
        #         This is equivalent to finding a collection of cliques in C that are edge-disjoint.
        #         Actually, it's simpler: We just need to partition a subset of E_CC into "blocks" assigned to v's.
        
        # Let's try a greedy approach with backtracking or just smart greedy:
        # 1. Start with all E_CC available.
        # 2. For each v in I, we want to pick a subset S_v of N(v) such that all edges in S_v are in Available_CC.
        #    To maximize benefit, we want S_v to be as large as possible.
        #    But picking a large S_v removes many edges from Available_CC, potentially hurting other v's.
        
        # Alternative View:
        # Consider the graph G_C induced by E_CC.
        # We want to select a set of cliques {S_v} such that:
        # 1. S_v ⊆ N(v)
        # 2. The edge sets of S_v are pairwise disjoint.
        # 3. Maximize sum(|S_v| choose 2) + sum(|S_v|) ? No, we just want to minimize total cliques.
        
        # Let's estimate the gain.
        # If we pick S_v of size m, we use 1 clique.
        # We cover m CI edges and m(m-1)/2 CC edges.
        # If we didn't pick S_v, we'd use m size-2 cliques for CI and m(m-1)/2 size-2 cliques for CC.
        # Total size-2 cliques avoided: m + m(m-1)/2 - 1.
        # We want to maximize this sum over all v.
        
        # Greedy Heuristic:
        # 1. Calculate for each v the "potential" of its neighbors.
        # 2. Process v's in some order.
        # 3. For each v, find the largest clique S in N(v) such that all edges in S are currently "available" (not used by previous S's).
        # 4. Assign S to v.
        # 5. Remove edges of S from Available_CC.
        
        # To make this better, we can use a weighted greedy:
        # Process v's with larger N(v) first? Or smaller?
        # Usually, larger N(v) has more options, so process smaller N(v) first to lock in their options?
        # Or process larger N(v) first to take the "best" edges?
        # Let's try processing by size of N(v) descending.
        
        Available_CC = set(E_CC)
        S_assignment = {} # v -> list of vertices in C
        
        # Sort I vertices by degree in C (descending)
        I_vertices = sorted(E_CI.keys(), key=lambda v: len(E_CI[v]), reverse=True)
        
        for v in I_vertices:
            if time.time() > deadline: break
            neighbors = E_CI[v]
            if not neighbors:
                continue
            
            # Find largest clique S in neighbors such that all edges in S are in Available_CC.
            # This is Max Clique in the subgraph induced by neighbors, where an edge (u1, u2) exists only if (u1, u2) in Available_CC.
            
            # Simple Greedy Max Clique:
            # 1. Order neighbors by degree in Available_CC (within neighbors).
            # 2. Try to add vertices one by one.
            
            # Build adjacency list for the "compatibility" graph on neighbors
            # Compat(u1, u2) = True if (u1, u2) in Available_CC
            
            # To speed up, we can just try a few random orderings or a specific greedy.
            # Given k<=30, even O(k^3) or O(k^4) is fine.
            
            # Let's try a simple greedy:
            # Sort neighbors by number of available edges to other neighbors.
            # Then iteratively add the vertex that maintains clique property.
            
            # Actually, since k is small, we can try to find a maximal clique using Bron-Kerbosch or similar?
            # No, standard Max Clique is hard. But k=30 is small enough for a simple backtracking if needed.
            # However, a greedy heuristic is likely sufficient for a good score.
            
            # Heuristic:
            # 1. Create list of candidates = neighbors.
            # 2. While candidates:
            #    Pick candidate u with max degree in (candidates ∩ Available_CC neighbors of u).
            #    Add u to S.
            #    Remove from candidates any w such that (u, w) not in Available_CC.
            #    This is a standard greedy max clique.
            
            S = []
            candidates = list(neighbors)
            
            # Sort candidates initially by degree in Available_CC (global) to break ties
            def deg_in_avail(u):
                count = 0
                for w in candidates:
                    if u != w and (min(u,w), max(u,w)) in Available_CC:
                        count += 1
                return count
            
            candidates.sort(key=deg_in_avail, reverse=True)
            
            while candidates:
                # Pick the first candidate (highest degree among remaining)
                u = candidates[0]
                S.append(u)
                
                # Filter candidates: keep only those connected to u in Available_CC
                new_candidates = []
                for w in candidates[1:]:
                    if (min(u,w), max(u,w)) in Available_CC:
                        new_candidates.append(w)
                candidates = new_candidates
                
            if S:
                S_assignment[v] = S
                # Remove edges in S from Available_CC
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        Available_CC.discard((min(S[i], S[j]), max(S[i], S[j])))
        
        # Now construct the cliques.
        cliques = []
        
        # 1. Add Type B cliques
        for v, S in S_assignment.items():
            if S:
                cliques.append(S + [v])
        
        # 2. Cover remaining CI edges with size-2 cliques
        for v in E_CI:
            neighbors = E_CI[v]
            S = S_assignment.get(v, [])
            S_set = set(S)
            for u in neighbors:
                if u not in S_set:
                    cliques.append([u, v])
        
        # 3. Cover remaining CC edges with size-2 cliques
        for u, v in Available_CC:
            cliques.append([u, v])
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)
            
    return results
# EVOLVE-BLOCK-END

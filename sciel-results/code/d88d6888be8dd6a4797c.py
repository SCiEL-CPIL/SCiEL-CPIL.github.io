"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

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
        
        # 1. Build Graph Structure
        # C = {0, ..., k-1}, I = {k, ..., n-1}
        # We need to cover all edges in E = E_CC U E_CI.
        
        # Represent edges as a set of frozensets or tuples for fast lookup
        edge_set = set(edges)
        
        # Adjacency for quick clique verification
        # Since k is small (<=30) and n is moderate, we can precompute adjacencies.
        # But for clique generation, we need to know if a subset is a clique.
        
        # Let's precompute the adjacency matrix or sets for the whole graph.
        # Vertices 0..n-1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def is_clique(vertices):
            # vertices is a list or set
            v_list = list(vertices)
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    if v_list[j] not in adj[v_list[i]]:
                        return False
            return True
            
        # 2. Generate Candidate Cliques
        # A clique partition must use cliques of size >= 2.
        # We want to find a set of cliques that covers all edges exactly once? 
        # Wait, the problem says "every edge must lie in exactly one listed clique".
        # This is a Clique Partition (or Edge-Disjoint Clique Cover).
        # The standard Set Cover allows overlapping. We need an Exact Cover or Partition.
        # However, if we pick a clique, we "remove" its edges.
        # If we pick two cliques that share an edge, that edge is covered twice -> Invalid.
        # So this is not standard Set Cover. It's a Packing/Covering problem with disjointness constraint on edges.
        
        # Let's re-evaluate.
        # If K1 and K2 are cliques, they can share at most 1 vertex. If they share 2 vertices, 
        # the edge between them is in both K1 and K2.
        # So the cliques must be edge-disjoint.
        
        # Strategy: 
        # 1. Identify all maximal cliques.
        # 2. Use a greedy approach: Pick the clique that covers the most "hard to cover" edges?
        #    Or simply: Pick the largest available clique, remove its edges, repeat.
        #    This is the "Greedy Maximal Clique Packing".
        
        # Let's try a smarter greedy:
        # 1. List all edges.
        # 2. While edges remain:
        #    a. Find all maximal cliques in the current graph (induced by remaining edges? No, 
        #       the graph structure is fixed, but we can only use edges not yet covered).
        #       Actually, a clique in the partition must be a clique in the original graph.
        #       And its edges must not be covered by previous cliques.
        #    b. Among all valid cliques (subsets of original vertices that form a clique in original graph 
        #       AND all their edges are currently uncovered), pick the one with maximum size (or weight).
        #    c. Add it to the partition.
        #    d. Remove its edges from the set of uncovered edges.
        
        # To make this efficient:
        # We can precompute all cliques of size >= 2? 
        # For n=90, k=30. 
        # Max cliques are likely small or involve many C vertices and one I vertex.
        # In split graphs, every maximal clique is of the form {v} U S where v in I and S subset N(v), 
        # OR a subset of C.
        
        # Let's generate candidate cliques:
        candidates = []
        
        # Type 1: Subsets of C that are cliques (all of C is a clique, so any subset is a clique).
        # But we only need to consider those that might be useful.
        # Actually, any subset of C is a clique.
        # Type 2: {v} U S where v in I, S subset N(v).
        
        # We will generate a pool of "good" cliques.
        # 1. For each v in I, find maximal cliques containing v.
        #    These are {v} U S where S is a maximal clique in G[N(v)]? No, N(v) is a subset of C, 
        #    and C is a clique. So N(v) is a clique. 
        #    So {v} U N(v) is a maximal clique (assuming v has no neighbors in I, which is true for split graphs).
        #    So the maximal cliques containing v are just {v} U N(v) and its subsets?
        #    No, if S is a subset of N(v), {v} U S is a clique.
        #    The maximal one is {v} U N(v).
        
        # 2. Maximal cliques not containing I vertices are subsets of C. 
        #    The maximal one is C itself.
        
        # So the "Maximal Cliques" of the graph are:
        # - C (if C is non-empty)
        # - {v} U N(v) for each v in I with N(v) non-empty.
        
        # However, we don't have to use maximal cliques. A smaller clique might be better 
        # if it leaves edges for a better combination?
        # Actually, larger cliques are generally better because they cover more edges per clique.
        # Greedy: Pick the largest available clique.
        
        # Let's implement "Greedy Largest Available Clique".
        
        uncovered_edges = set(edge_set)
        partition = []
        
        # Pre-compute all potential cliques?
        # Number of vertices in C is k <= 30.
        # Number of vertices in I is n-k.
        
        # Potential cliques:
        # 1. Any subset of C. (Too many: 2^k).
        # 2. {v} U S, S subset N(v). (2^{|N(v)|} per v).
        
        # We can't enumerate all. We need to generate them on the fly or use a heuristic.
        
        # Heuristic:
        # 1. Start with the set of all maximal cliques.
        # 2. Iteratively pick the clique that covers the most uncovered edges.
        # 3. If a maximal clique has all its edges covered, it's not a candidate.
        # 4. But maybe a subset of a maximal clique is better?
        #    If K is a maximal clique, and some edges in K are covered, we can only use a subset K' of K 
        #    such that all edges in K' are uncovered.
        #    To maximize coverage, we should pick the largest subset of K with all edges uncovered.
        
        # Algorithm:
        # 1. Generate list of Maximal Cliques.
        # 2. While uncovered_edges:
        #    a. For each maximal clique M:
        #       - Find the largest subset S of M such that all edges induced by S are in uncovered_edges.
        #         This is equivalent to finding the largest clique in the graph H_M induced by vertices of M, 
        #         where edges of H_M are exactly the uncovered edges between vertices in M.
        #    b. Select the M that yields the largest such S (or highest weight).
        #    c. Add S to partition.
        #    d. Remove edges of S from uncovered_edges.
        
        # This seems robust.
        
        # Step 1: Generate Maximal Cliques
        max_cliques = []
        
        # Clique C
        if k > 1:
            max_cliques.append(list(range(k)))
            
        # Cliques {v} U N(v)
        # We need N(v) for each v in I.
        # We already have adj.
        for v in range(k, n):
            neighbors = [u for u in range(k) if u in adj[v]]
            if neighbors:
                # The clique is [v] + neighbors
                # But is it maximal? 
                # Yes, because v has no neighbors in I, and C is a clique.
                # Any other vertex u in C not in neighbors is not adjacent to v.
                # So {v} U N(v) is maximal.
                max_cliques.append([v] + neighbors)
                
        # Note: Some maximal cliques might be subsets of others?
        # No, C is disjoint from I. 
        # {v} U N(v) contains v (in I). C does not contain v.
        # So they are distinct maximal cliques.
        
        # Step 2: Greedy Selection
        # To speed up "Find largest subset of M with all edges uncovered",
        # we can use a simple greedy max-clique on the subgraph induced by M using uncovered edges.
        
        # Optimization: 
        # The "uncovered edges" graph changes dynamically.
        
        while uncovered_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges as size-2 cliques
                for e in uncovered_edges:
                    partition.append(list(e))
                break
                
            best_subset = []
            best_size = 0
            
            for M in max_cliques:
                # M is a list of vertices.
                # We want the largest subset S of M such that all edges in S are in uncovered_edges.
                # Let's construct the subgraph on M with only uncovered edges.
                # Since M is a clique in the original graph, the only constraint is which edges are still uncovered.
                
                # Greedy Max Clique on M with respect to uncovered_edges
                # Vertices: M
                # Edge (u,v) exists if (min(u,v), max(u,v)) in uncovered_edges
                
                # Simple Greedy:
                # 1. Candidates = list(M)
                # 2. Sort candidates by degree in the subgraph?
                # 3. Pick one, filter others.
                
                candidates = list(M)
                # Sort by degree in uncovered subgraph
                def deg_in_sub(u):
                    cnt = 0
                    for w in candidates:
                        if u == w: continue
                        if (min(u,w), max(u,w)) in uncovered_edges:
                            cnt += 1
                    return cnt
                    
                # To avoid O(|M|^2) for every M every iteration, we might need to be careful.
                # But |M| <= k+1 <= 31. And number of max cliques <= n-k+1 <= 91.
                # Iterations: at most number of edges / min_clique_size.
                # Edges ~ n^2/4 ~ 2000. Min clique size 2. So ~1000 iterations worst case.
                # 1000 * 91 * 31^2 ~ 9 million ops. Should be fine in Python if optimized.
                
                # Let's try a simpler heuristic for finding the subset:
                # Just try to add vertices one by one?
                # Or since M is small, we can just check all subsets? 2^31 is too big.
                # Greedy is safer.
                
                # Refine Greedy:
                # Start with all vertices in M.
                # Repeatedly remove the vertex with the fewest connections to other candidates in uncovered_edges?
                # No, we want to KEEP a large clique.
                # Standard Greedy Max Clique:
                # 1. Order vertices by degree in subgraph (descending).
                # 2. Iterate through ordered vertices.
                # 3. If vertex is connected to all currently selected vertices, add it.
                
                selected = []
                # We need the subgraph adjacency within M
                # Let's just use the check function.
                
                # Sort M by degree in uncovered subgraph
                # This requires knowing the edges.
                # Let's just use a random shuffle or fixed order to avoid bias?
                # Actually, degree is a good heuristic.
                
                # To compute degree fast, we can just count.
                # Since M is small, O(|M|^2) is fine.
                
                # Let's implement a specific greedy for this small subgraph
                cand_list = list(M)
                # Sort by degree in uncovered_edges within cand_list
                def local_deg(u):
                    return sum(1 for w in cand_list if u != w and (min(u,w), max(u,w)) in uncovered_edges)
                
                cand_list.sort(key=local_deg, reverse=True)
                
                for u in cand_list:
                    # Check if u is connected to all in selected
                    is_conn = True
                    for s in selected:
                        if (min(u,s), max(u,s)) not in uncovered_edges:
                            is_conn = False
                            break
                    if is_conn:
                        selected.append(u)
                        
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_subset = selected
                    
            if not best_subset:
                # Should not happen if uncovered_edges is not empty and max_cliques cover the graph
                # But just in case, pick an arbitrary edge
                e = next(iter(uncovered_edges))
                best_subset = list(e)
                
            # Add best_subset to partition
            if len(best_subset) >= 2:
                partition.append(best_subset)
                # Remove its edges
                for i in range(len(best_subset)):
                    for j in range(i+1, len(best_subset)):
                        u, v = best_subset[i], best_subset[j]
                        uncovered_edges.discard((min(u,v), max(u,v)))
            else:
                # If best_subset size < 2, something is wrong. 
                # This implies an edge is isolated from any clique in max_cliques?
                # Impossible in a connected graph covered by max cliques.
                # Unless the edge is not in any max clique? 
                # Every edge is in some max clique.
                # So this shouldn't happen.
                pass
                
        return partition

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            # Ensure types are correct
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                if len(clean_clique) >= 2:
                    clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END

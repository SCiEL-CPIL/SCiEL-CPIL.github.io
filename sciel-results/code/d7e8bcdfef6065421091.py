"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def greedy_clique_partition(edges_set):
        """
        Greedy approach to partition a set of edges into cliques.
        Repeatedly find a large clique in the remaining edge graph and remove its edges.
        """
        if not edges_set:
            return []
        
        rem_edges = set(edges_set)
        cliques = []
        
        while rem_edges:
            # Compute degrees
            degrees = defaultdict(int)
            for u, v in rem_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            # Pick vertex with max degree
            max_v = max(degrees, key=degrees.get)
            
            # Find neighbors of max_v in rem_edges
            neighbors = set()
            for u, v in rem_edges:
                if u == max_v:
                    neighbors.add(v)
                elif v == max_v:
                    neighbors.add(u)
            
            # Sort neighbors by degree to prioritize dense areas
            candidate_neighbors = list(neighbors)
            candidate_neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            clique_vertices = [max_v]
            
            for cand in candidate_neighbors:
                # Check if cand is connected to all existing clique_vertices
                is_connected = True
                for cv in clique_vertices:
                    if cv == cand:
                        continue
                    if (min(cv, cand), max(cv, cand)) not in rem_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.append(cand)
            
            cliques.append(sorted(clique_vertices))
            
            # Remove all edges within this clique from rem_edges
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    u, w = clique_vertices[i], clique_vertices[j]
                    edge = (min(u, w), max(u, w))
                    if edge in rem_edges:
                        rem_edges.remove(edge)
                        
            # Check deadline
            if time.time() > deadline - 5:
                # Fallback: cover each remaining edge individually
                for e in rem_edges:
                    cliques.append([e[0], e[1]])
                rem_edges.clear()
                break

        return cliques

    def solve_split(n, k, p, seed):
        """
        Solve the split graph clique partition problem.
        Strategy:
        1. Identify the clique K (vertices 0..k-1) and independent set I (vertices k..n-1).
        2. For each I vertex v, we can form a clique {v} ∪ S where S ⊆ N(v) ∩ K.
           Since K is a clique, any subset S of N(v) forms a clique with v.
        3. We want to maximize the "savings" from forming such cliques.
           A clique of size m saves (m*(m-1)/2 - 1) edges compared to individual 2-cliques.
           However, using K-edges in an I-clique means they can't be used in K-only cliques.
           
        Greedy heuristic:
        - Sort I vertices by degree descending.
        - For each I vertex v, form a clique with as many K-neighbors as possible,
          prioritizing K-vertices that have many remaining K-edges to "absorb".
        - Actually, a simpler and effective heuristic:
          For each I vertex v, let S_v = N(v). Form clique {v} ∪ S_v.
          This covers all cross edges from v and all K-edges within S_v.
          Remove these K-edges from the available pool.
        - After processing all I vertices, cover remaining K-edges with K-only cliques.
        - Cover any remaining uncovered cross edges (shouldn't happen if we include all neighbors).
        
        But wait: if we include ALL neighbors, we might be "wasting" K-edges that could have been part of a larger K-only clique.
        However, including all neighbors in {v} ∪ S_v is always beneficial because:
        - It covers |S_v| cross edges with 1 clique (instead of |S_v| 2-cliques).
        - It covers |S_v|*(|S_v|-1)/2 K-edges with that same clique.
        - The "cost" is that these K-edges can't be used in K-only cliques.
        - But K-only cliques also cover multiple K-edges, so we need to balance.
        
        Alternative: Process I vertices in order of degree descending.
        For v, let S_v = N(v). The clique {v} ∪ S_v has size |S_v| + 1.
        Savings = (|S_v|+1)*|S_v|/2 - 1.
        The K-edges consumed are all pairs within S_v.
        
        Let's try: For each I vertex v (sorted by degree desc), form clique {v} ∪ N(v).
        Remove all K-edges within N(v) from the available K-edge set.
        Then cover remaining K-edges with greedy clique partition.
        
        This is simple and likely effective.
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K-vertices and cross edges
        cross_edges = []
        neighbors_in_K = defaultdict(set)
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_K[v].add(u)
                cross_edges.append((u, v))
            elif u >= k and v < k:
                neighbors_in_K[u].add(v)
                cross_edges.append((v, u))
        
        # All K-edges
        all_K_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_K_edges.add((u, w))
        
        available_K_edges = set(all_K_edges)
        cliques = []
        
        # Sort I vertices by degree descending
        i_vertices = [v for v in range(k, n) if neighbors_in_K[v]]
        i_vertices.sort(key=lambda v: len(neighbors_in_K[v]), reverse=True)
        
        for v in i_vertices:
            S_v = neighbors_in_K[v]
            if len(S_v) >= 1:
                # Form clique {v} ∪ S_v
                clique = sorted(list(S_v) + [v])
                cliques.append(clique)
                # Remove K-edges within S_v from available_K_edges
                S_list = list(S_v)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u, w = S_list[i], S_list[j]
                        e = (min(u, w), max(u, w))
                        if e in available_K_edges:
                            available_K_edges.remove(e)
        
        # Cover remaining K-edges
        if available_K_edges:
            k_cliques = greedy_clique_partition(available_K_edges)
            cliques.extend(k_cliques)
        
        # Cover any cross edges not covered (should be none if we included all neighbors)
        covered_cross = set()
        for v in i_vertices:
            for u in neighbors_in_K[v]:
                covered_cross.add((min(u, v), max(u, v)))
        
        for u, v in cross_edges:
            e = (min(u, v), max(u, v))
            if e not in covered_cross:
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline - 10:
            # Fallback for remaining instances: trivial solution
            edges = instance(n, k, p, seed)
            cliques = [[u, v] for u, v in edges]
            results.append(cliques)
            continue
        
        cliques = solve_split(n, k, p, seed)
        
        # Convert to pure Python types
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
# EVOLVE-BLOCK-END

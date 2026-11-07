"""Combined approach: per-I-vertex greedy clique extraction + global K-edge clique cover + Local Search."""
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

    def norm_edge(u, v):
        return (min(u, v), max(u, v))

    def bron_kerbosch_max_clique(adj_sets, vertices):
        """
        Find a maximum clique in the graph defined by adj_sets (list of sets) on vertices.
        Uses Bron-Kerbosch with pivot for efficiency.
        Returns the largest clique found as a list of vertex indices.
        """
        if not vertices:
            return []

        best = []
        vertices_set = set(vertices)

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return

            # Pivot selection: choose vertex from P∪X with max degree in P
            pivot_candidates = list(P | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda u: len(P & adj_sets[u]))
                candidates = [v for v in P if v not in adj_sets[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj_sets[v]
                new_X = X & adj_sets[v]
                P.remove(v)
                X.add(v)
                bk(new_R, new_P, new_X)
                P.add(v)
                X.remove(v)

        bk(set(), vertices_set, set())
        return best

    def greedy_max_clique(adj_sets, vertices):
        """
        Greedy approach to find a large clique.
        Start with the vertex of highest degree, then iteratively add vertices
        that are connected to all current clique members.
        """
        if not vertices:
            return []
        
        v_list = list(vertices)
        if not v_list:
            return []
        
        # Start with highest degree vertex
        start = max(v_list, key=lambda v: len(adj_sets[v]))
        clique = {start}
        remaining = set(v_list) - {start}
        
        while remaining:
            # Find vertices in remaining that are connected to all in clique
            candidates = [v for v in remaining if all(c in adj_sets[v] for c in clique)]
            if not candidates:
                break
            # Pick the one with highest degree among candidates
            best = max(candidates, key=lambda v: len(adj_sets[v]))
            clique.add(best)
            remaining.discard(best)
        
        return list(clique)

    def cover_edges_with_cliques(edge_set, vertex_range, rng=None):
        """
        Cover a set of edges with minimum number of cliques.
        Uses greedy: repeatedly find a large clique in the remaining edge graph.
        """
        cliques = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if time.time() > deadline - 2:
                # Fallback: cover each edge individually
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Build adjacency for the remaining edge graph
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            v_list = list(active_vertices)
            m = len(v_list)
            idx_map = {v: i for i, v in enumerate(v_list)}
            
            # Build adjacency sets
            adj_sets = [set() for _ in range(m)]
            for u, v in remaining_edges:
                i = idx_map[u]
                j = idx_map[v]
                adj_sets[i].add(j)
                adj_sets[j].add(i)
            
            # Find a large clique
            if m <= 25:
                best_local = bron_kerbosch_max_clique(adj_sets, list(range(m)))
            else:
                best_local = greedy_max_clique(adj_sets, list(range(m)))
            
            if not best_local:
                # Take any single edge
                u, v = next(iter(remaining_edges))
                S = [u, v]
            else:
                S = [v_list[idx] for idx in best_local]
            
            cliques.append(S)
            
            # Remove all edges within S from remaining_edges
            S_set = set(S)
            edges_to_remove = set()
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    e = norm_edge(S[i], S[j])
                    if e in remaining_edges:
                        edges_to_remove.add(e)
            remaining_edges -= edges_to_remove
        
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        neighbors_in_k = {v: set() for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_k[v].add(u)
            elif v < k and u >= k:
                neighbors_in_k[u].add(v)
        
        used_edges = set()
        cliques = []
        
        # Process each v in I: greedily extract cliques containing v
        # Sort I vertices by decreasing degree to cover more edges first
        i_vertices = sorted(range(k, n), key=lambda v: -len(neighbors_in_k[v]))
        
        for v in i_vertices:
            if time.time() > deadline - 2:
                # Fallback: cover remaining cross edges with pairs
                for u in neighbors_in_k[v]:
                    if norm_edge(u, v) not in used_edges:
                        cliques.append([v, u])
                        used_edges.add(norm_edge(u, v))
                continue
            
            N_v = set(neighbors_in_k[v])
            remaining = list(N_v)
            
            while remaining:
                if time.time() > deadline - 2:
                    for u in remaining:
                        e = norm_edge(v, u)
                        if e not in used_edges:
                            cliques.append([v, u])
                            used_edges.add(e)
                    break
                
                m = len(remaining)
                if m == 1:
                    u = remaining[0]
                    e = norm_edge(v, u)
                    if e not in used_edges:
                        cliques.append([v, u])
                        used_edges.add(e)
                    remaining = []
                    continue
                
                # Build adjacency for the "available" graph on remaining
                # Two K-vertices u, w are connected if edge (u,w) is NOT in used_edges
                rem_set = set(remaining)
                adj_avail = [set() for _ in range(m)]
                idx_map = {v_idx: i for i, v_idx in enumerate(remaining)}
                
                for i in range(m):
                    for j in range(i + 1, m):
                        u, w = remaining[i], remaining[j]
                        e = norm_edge(u, w)
                        if e not in used_edges:
                            adj_avail[i].add(j)
                            adj_avail[j].add(i)
                
                # Find a large clique in adj_avail
                if m <= 25:
                    best_local = bron_kerbosch_max_clique(adj_avail, list(range(m)))
                else:
                    best_local = greedy_max_clique(adj_avail, list(range(m)))
                
                if not best_local:
                    best_local = [0]
                
                S = [remaining[idx] for idx in best_local]
                
                # Add clique [v] + S
                cliques.append([v] + S)
                
                # Mark all edges in this clique as used
                for u in S:
                    used_edges.add(norm_edge(v, u))
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        used_edges.add(norm_edge(S[i], S[j]))
                
                # Remove S from remaining
                S_set = set(S)
                remaining = [x for x in remaining if x not in S_set]
        
        # Cover remaining K-edges
        E_rem = set()
        for u in range(k):
            for w in range(u + 1, k):
                if norm_edge(u, w) not in used_edges:
                    E_rem.add(norm_edge(u, w))
        
        if E_rem:
            k_cliques = cover_edges_with_cliques(E_rem, range(k))
            cliques.extend(k_cliques)
        
        # Local Search: Try to improve the solution by re-covering subsets of cliques
        # This is a heuristic: remove a few cliques, re-cover their edges, see if it's better.
        best_cliques = cliques
        best_count = len(cliques)
        
        # Map edge to clique index for quick lookup? Or just recompute.
        # Since n is small, we can afford some recomputation.
        
        rng_local = Random(42)
        
        while time.time() < deadline - 5:
            # Randomly select a subset of cliques to "relax"
            # We select cliques that share vertices or are "small"
            # For simplicity, select a random small number of cliques
            num_to_relax = rng_local.randint(1, 5)
            if len(best_cliques) < num_to_relax:
                break
                
            indices_to_relax = rng_local.sample(range(len(best_cliques)), num_to_relax)
            
            # Collect edges from these cliques
            relax_edges = set()
            for idx in indices_to_relax:
                c = best_cliques[idx]
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        relax_edges.add(norm_edge(c[i], c[j]))
            
            # Remove these cliques from the list
            # Create a new list excluding these indices
            new_cliques = [c for i, c in enumerate(best_cliques) if i not in indices_to_relax]
            
            # Re-cover the relax_edges using the greedy method
            # Note: These edges were previously covered, but now we treat them as "available".
            # We must ensure we don't double cover edges already in new_cliques.
            # However, relax_edges are disjoint from edges in new_cliques by construction 
            # (since each edge is in exactly one clique in the valid solution).
            
            # But wait, the greedy method assumes a set of edges to cover.
            # We just call cover_edges_with_cliques on relax_edges.
            # This will find cliques within the subgraph induced by relax_edges.
            
            new_relax_cliques = cover_edges_with_cliques(relax_edges, range(n))
            
            candidate_cliques = new_cliques + new_relax_cliques
            candidate_count = len(candidate_cliques)
            
            if candidate_count < best_count:
                best_cliques = candidate_cliques
                best_count = candidate_count
                # Reset search if improved? Or continue?
                # Continue to see if we can improve further.
            elif candidate_count == best_count and rng_local.random() < 0.1:
                # Accept equal solutions with small probability to escape local optima
                best_cliques = candidate_cliques
        
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_one(n, k, p, seed)
        results.append(cliques)
    
    return results

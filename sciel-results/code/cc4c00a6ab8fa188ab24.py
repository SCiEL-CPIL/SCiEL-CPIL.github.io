import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def trivial_partition(edges):
        return [list(e) for e in edges]

    def find_maximal_clique_in_neighbors(v, neighbors, remaining_edges):
        # neighbors is a set of vertices connected to v in remaining_edges
        # We want a maximal clique in the subgraph induced by neighbors
        # using remaining_edges as the edge set
        
        candidates = set(neighbors)
        clique = set()
        
        while candidates:
            # Pick a candidate, e.g., one with max degree in the induced subgraph of candidates
            # For simplicity and speed, just pick an arbitrary one, or the first one
            # Better: pick one with max connections to other candidates to grow clique fast
            best_u = None
            max_conn = -1
            for u in candidates:
                conn = 0
                for w in candidates:
                    if u == w: continue
                    if (min(u, w), max(u, w)) in remaining_edges:
                        conn += 1
                # Also check connection to existing clique members
                valid = True
                for x in clique:
                    if (min(u, x), max(u, x)) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    continue
                if conn > max_conn:
                    max_conn = conn
                    best_u = u
            
            if best_u is None:
                # No valid candidate found? This shouldn't happen if we started with valid neighbors
                # But if clique is empty, any neighbor is valid.
                # If clique is not empty, we need to check validity.
                # The loop above checks validity. If none are valid, we break?
                # Actually, if we can't add any, the clique is maximal.
                break
            
            u = best_u
            clique.add(u)
            
            # Update candidates: must be connected to u
            new_candidates = set()
            for w in candidates:
                if w == u: continue
                if (min(u, w), max(u, w)) in remaining_edges:
                    new_candidates.add(w)
            candidates = new_candidates
            
        return clique

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline:
                return trivial_partition(edges)
            
            # Find vertex with max degree in remaining_edges
            degrees = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees or max(degrees.values()) == 0:
                break
                
            v = max(degrees, key=degrees.get)
            
            # Find neighbors of v in remaining_edges
            neighbors = set()
            for u, w in remaining_edges:
                if u == v:
                    neighbors.add(w)
                elif w == v:
                    neighbors.add(u)
            
            if not neighbors:
                # v is isolated in remaining graph? Should not happen if degree > 0
                break
            
            # Find maximal clique in neighbors
            clique_neighbors = find_maximal_clique_in_neighbors(v, neighbors, remaining_edges)
            
            # The full clique includes v
            full_clique = set(clique_neighbors)
            full_clique.add(v)
            
            # Remove all edges within full_clique from remaining_edges
            edges_to_remove = set()
            clique_list = list(full_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    u, w = clique_list[i], clique_list[j]
                    edge = (min(u, w), max(u, w))
                    if edge in remaining_edges:
                        edges_to_remove.add(edge)
            
            remaining_edges.difference_update(edges_to_remove)
            
            if len(full_clique) >= 2:
                cliques.append(sorted(list(full_clique)))
            else:
                # Should not happen as we pick v with deg > 0, so neighbors not empty,
                # and clique_neighbors might be empty if no edges among neighbors?
                # If clique_neighbors is empty, full_clique is {v}. Size 1.
                # But we need cliques of size >= 2.
                # If we can't form a clique of size >= 2, we must take an edge.
                # However, if v has a neighbor u, (v,u) is an edge.
                # If no other vertex connects to both v and u, the clique is just {v,u}.
                # My find_maximal_clique_in_neighbors might return empty if it fails to add anyone?
                # Let's trace: candidates = neighbors.
                # If we pick u, clique={u}.
                # If we can't add anyone else, clique={u}.
                # full_clique = {v, u}. Size 2.
                # So it should always be size >= 2.
                pass

        # Fallback: if for some reason we didn't cover all edges (e.g. logic error),
        # ensure we return a valid partition.
        # Check if all edges are covered.
        covered = set()
        for c in cliques:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, w = c[i], c[j]
                    covered.add((min(u, w), max(u, w)))
        
        if covered != edges:
             # Return trivial partition to ensure validity
             return trivial_partition(edges)
             
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Generate trivial for remaining
            edges = generate_graph(n, p, seed)
            results.append(trivial_partition(edges))
        else:
            results.append(solve_instance(n, p, seed))
            
    return results

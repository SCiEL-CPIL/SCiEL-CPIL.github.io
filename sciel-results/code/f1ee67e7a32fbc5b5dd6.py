"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return edges, cross_edges

    def solve_instance(n, k, p, seed, deadline):
        internal_edges, cross_edges = instance(n, k, p, seed)
        
        # Build set of internal K edges that are still uncovered
        # Initially all internal edges are uncovered
        uncovered_internal = set()
        for (u, v) in internal_edges:
            uncovered_internal.add((min(u, v), max(u, v)))
        
        # Build neighbor sets for I-vertices
        # neighbors[v] = set of u in K such that (u, v) is a cross edge
        neighbors = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            neighbors[v].add(u)
        
        cliques = []
        
        # Process I-vertices in order of decreasing degree (greedy heuristic)
        # Vertices with more connections to K can form larger cliques
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in I_vertices:
            if time.time() > deadline - 1:
                # Fallback: just use singleton cliques for remaining
                for u in neighbors[v]:
                    cliques.append([u, v])
                continue
            
            N_v = neighbors[v]
            remaining = set(N_v)
            
            # Greedily find largest clique in the subgraph induced by 'remaining'
            # where edges are those still in uncovered_internal
            while remaining:
                if time.time() > deadline - 1:
                    for u in remaining:
                        cliques.append([u, v])
                    break
                
                # Find a clique within 'remaining' such that all edges among them
                # are in uncovered_internal
                # Greedy: pick vertex with max "available" degree, then restrict
                # to its neighbors that also have edges to all previously picked
                
                # Compute available degree: number of other vertices in 'remaining'
                # that are connected to it via an edge in uncovered_internal
                best_clique = [max(remaining, key=lambda x: sum(
                    1 for y in remaining if y != x and (min(x,y), max(x,y)) in uncovered_internal
                ))]
                
                # Try to extend the clique
                changed = True
                while changed:
                    changed = False
                    candidates = [x for x in remaining if x not in best_clique]
                    for x in candidates:
                        # Check if x can be added: edge (x, y) must be in uncovered_internal
                        # for all y in best_clique
                        can_add = True
                        for y in best_clique:
                            if (min(x, y), max(x, y)) not in uncovered_internal:
                                can_add = False
                                break
                        if can_add:
                            best_clique.append(x)
                            changed = True
                            break
                
                if len(best_clique) >= 2:
                    # Add clique {v} ∪ best_clique
                    clique = [v] + best_clique
                    cliques.append(clique)
                    
                    # Remove internal edges covered by this clique
                    for i in range(len(best_clique)):
                        for j in range(i + 1, len(best_clique)):
                            u, w = best_clique[i], best_clique[j]
                            uncovered_internal.discard((min(u, w), max(u, w)))
                    
                    # Remove best_clique vertices from remaining
                    remaining -= set(best_clique)
                else:
                    # No clique of size >= 2, use singleton
                    u = max(remaining)
                    cliques.append([u, v])
                    remaining.discard(u)
        
        # Cover remaining internal K edges
        if uncovered_internal:
            if time.time() > deadline - 1:
                # Fallback: add all remaining vertices as one clique
                V_rem = set()
                for (u, w) in uncovered_internal:
                    V_rem.add(u)
                    V_rem.add(w)
                if V_rem:
                    cliques.append(sorted(V_rem))
            else:
                # Greedily cover remaining internal edges with cliques
                remaining_edges = set(uncovered_internal)
                while remaining_edges:
                    if time.time() > deadline - 1:
                        V_rem = set()
                        for (u, w) in remaining_edges:
                            V_rem.add(u)
                            V_rem.add(w)
                        if V_rem:
                            cliques.append(sorted(V_rem))
                        break
                    
                    # Find a maximal clique in the graph defined by remaining_edges
                    # Start with an edge
                    first_edge = next(iter(remaining_edges))
                    u, w = first_edge
                    clique = [u, w]
                    
                    # Try to extend
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x in clique:
                                continue
                            can_add = True
                            for y in clique:
                                if (min(x, y), max(x, y)) not in remaining_edges:
                                    can_add = False
                                    break
                            if can_add:
                                clique.append(x)
                                changed = True
                                break
                    
                    cliques.append(clique)
                    
                    # Remove edges covered by this clique
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            a, b = clique[i], clique[j]
                            remaining_edges.discard((min(a, b), max(a, b)))
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    results = []
    for spec in specs:
        if time.time() > deadline - 5:
            # Emergency fallback: simple solution
            n, k, p, seed = spec
            internal_edges, cross_edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in cross_edges:
                cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(solve_instance(*spec, deadline))
    
    # Ensure proper types
    final = []
    for cliques in results:
        final_cliques = []
        for clique in cliques:
            final_cliques.append([int(v) for v in clique])
        final.append(final_cliques)
    
    return final

# EVOLVE-BLOCK-END

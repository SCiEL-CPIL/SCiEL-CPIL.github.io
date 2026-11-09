"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def generate_edges(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_pivot(R, P, X, adj, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot: choose vertex in P ∪ X with max degree in P
        if P or X:
            pivot = max(P | X, key=lambda v: len(adj[v] & P))
        else:
            return
        candidates = P - adj[pivot]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch_pivot(new_R, new_P, new_X, adj, cliques)
            P.remove(v)
            X.add(v)
    
    def enumerate_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        cliques = []
        bron_kerbosch_pivot(set(), set(range(n)), set(), adj, cliques)
        return [list(c) for c in cliques if len(c) >= 2]
    
    def greedy_hitting_set(cliques, n):
        # Build incidence: vertex -> list of clique indices it hits
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        # Track which cliques are still uncovered
        uncovered = set(range(len(cliques)))
        selected = set()
        
        while uncovered:
            # Find vertex that covers most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            selected.add(best_v)
            # Remove all cliques containing best_v from uncovered
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        
        return selected
    
    def local_search(selected, cliques, n, time_limit):
        selected = set(selected)
        start = time.time()
        
        # Precompute: for each vertex, which cliques it covers
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            
            # Try deletions: remove a vertex if all cliques are still covered
            for v in list(selected):
                if time.time() - start > time_limit:
                    break
                # Check if removing v still covers all cliques
                ok = True
                for ci in vertex_cliques[v]:
                    # Check if clique ci has another vertex in selected
                    clique = cliques[ci]
                    has_other = False
                    for u in clique:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        ok = False
                        break
                if ok:
                    selected.discard(v)
                    improved = True
            
            # Try 2-opt swaps: remove v1, add v2 if it reduces size or keeps same
            for v1 in list(selected):
                if time.time() - start > time_limit:
                    break
                # Find cliques that would become uncovered if v1 removed
                at_risk = []
                for ci in vertex_cliques[v1]:
                    clique = cliques[ci]
                    has_other = False
                    for u in clique:
                        if u != v1 and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        at_risk.append(ci)
                
                if not at_risk:
                    # v1 is redundant, already handled by deletion
                    continue
                
                # Find v2 not in selected that covers all at_risk cliques
                for v2 in range(n):
                    if v2 in selected:
                        continue
                    covers_all = all(v2 in cliques[ci] for ci in at_risk)
                    if covers_all:
                        # Check if this swap actually helps (shouldn't since we try deletions first)
                        # Only swap if it might help in future iterations
                        pass
            
            # Try additions of vertices that enable removals
            # This is more complex, skip for now
        
        return selected
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Return best effort: all touched vertices
            edges = generate_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            out.append([])
            continue
        
        # Greedy hitting set
        selected = greedy_hitting_set(cliques, n)
        
        # Local search with time limit
        remaining = max(0, deadline - time.time() - 5)  # leave 5s buffer
        if remaining > 1:
            selected = local_search(selected, cliques, n, remaining)
        
        out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END

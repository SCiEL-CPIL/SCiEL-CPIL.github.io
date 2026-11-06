import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def random_graph(n, p, seed):
        # Replicate the seed program's graph generation exactly
        import random
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj, n):
        # Bron-Kerbosch with pivot for maximal cliques
        # Returns list of maximal cliques (as tuples of vertices)
        # We only care about cliques of size >= 2
        
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P U X
            # u with max degree in P
            u = None
            max_deg = -1
            for v in P.union(X):
                deg = len(adj[v].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    u = v
            
            # Branch on vertices in P not adjacent to u
            # If u is None, branch on all P
            candidates = P.difference(adj[u]) if u is not None else P.copy()
            
            for v in list(candidates):
                # v is in R
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj, edges = random_graph(n, p, seed)
        
        # If no edges, no cliques of size >= 2, return empty
        if not edges:
            return []

        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        if not cliques:
            return []

        # Greedy Set Cover
        # uncovered_cliques: set of frozensets
        uncovered = set(cliques)
        selected = []
        
        # Precompute for each vertex, the list of cliques it is in
        # This is useful for the reduction phase
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Track which cliques are covered by which selected vertices
        # Actually, for greedy, we just need to know which are uncovered.
        
        while uncovered:
            if time.time() > deadline:
                # Fallback: return all vertices involved in any edge to ensure validity
                # This is the seed behavior, guaranteed valid
                all_verts = set()
                for e in edges:
                    all_verts.add(e[0])
                    all_verts.add(e[1])
                return sorted(all_verts)

            best_v = -1
            best_count = -1
            
            # We can optimize by only considering vertices in the union of uncovered cliques
            # But n is small (<=40), so checking all is fine.
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for c in uncovered:
                    if v in c:
                        count += 1
                        if count > best_count:
                            # Early break if we know we can't beat it? 
                            # Max possible is len(uncovered)
                            pass 
                
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                # Should not happen if uncovered is not empty and vertices are in cliques
                # But just in case, pick any vertex from an uncovered clique
                for c in uncovered:
                    best_v = next(iter(c))
                    break
            
            selected.append(best_v)
            
            # Remove all cliques containing best_v from uncovered
            new_uncovered = set()
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.add(c)
            uncovered = new_uncovered
        
        # Reduction Phase
        # Try to remove each selected vertex if it's not essential
        # A vertex is essential if it is the ONLY selected vertex in at least one clique.
        # We can represent the coverage: for each clique, how many selected vertices are in it?
        
        # Map clique index to count of selected vertices
        # But we need to know WHICH selected vertices are in it to remove them.
        
        # Let's use a list of selected vertices.
        # For each clique, we need to know the count of selected vertices in it.
        
        # Rebuild coverage counts
        # Since n is small, we can just check.
        
        # Optimization: Only check vertices that are in at least one clique.
        # We iterate through selected vertices.
        
        # To make removal efficient, we maintain for each clique the set of selected vertices in it.
        # cliques are frozensets.
        
        # Let's just do a simple loop.
        # For each v in selected:
        #   Check if there is any clique c such that (c & selected_set) == {v}
        #   If no such clique exists, v is not essential. Remove it.
        
        # However, removing v might make another vertex essential? No, removing a vertex
        # only decreases counts. It cannot make a vertex *more* essential in the sense of being
        # the *only* one. It can only make a clique *less* covered.
        # Wait, if we remove v, a clique that had {v, w} now has {w}. w becomes essential.
        # But w was already in the solution. We are checking if v is essential.
        # If v is not essential, we remove it.
        
        # We should iterate and remove if possible.
        
        selected_set = set(selected)
        
        # We need to be careful: if we remove v, we update the state.
        # It's safer to iterate until no changes or just one pass?
        # One pass is usually sufficient for this greedy structure, but multiple passes can help.
        # Given time limits, one pass is safe.
        
        # Actually, let's check each vertex.
        # If we remove a vertex, we don't re-evaluate previously checked vertices in the same pass.
        # This is a standard "lazy" reduction.
        
        # To be more robust, we can check in reverse order of addition (last added first)?
        # Or just any order.
        
        for v in selected[:]: # Iterate over a copy
            # Check if v is essential
            essential = False
            for c in cliques:
                # Count how many vertices from selected_set are in c
                # If count == 1 and that vertex is v, then v is essential.
                # If count > 1, v is not essential for this clique.
                # If count == 0, this clique is uncovered! (Should not happen if greedy was correct)
                
                # Optimization: Precompute intersection size?
                # n is small.
                inter = c & selected_set
                if len(inter) == 1 and v in inter:
                    essential = True
                    break
            if not essential:
                selected_set.remove(v)
                selected.remove(v)
        
        return sorted(selected_set)

    results = []
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    for n, p, seed in instances:
        if time.time() > deadline:
             # Fallback for remaining instances
             adj, edges = random_graph(n, p, seed)
             all_verts = set()
             for e in edges:
                 all_verts.add(e[0])
                 all_verts.add(e[1])
             results.append(sorted(all_verts))
             continue
             
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results

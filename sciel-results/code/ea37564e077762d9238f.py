import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 150.0 - 10.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        cliques = []
        
        # C is [0, k), I is [k, n)
        # Step 1: Cover all cross edges and some C edges using I vertices
        # For each v in I, clique is {v} U N_C(v)
        # Track which edges in C are covered
        # Edge (u, w) in C is covered if there exists v in I such that u in N_C(v) and w in N_C(v)
        
        # Precompute N_C(v) for v in I
        N_C = {}
        for v in range(k, n):
            N_C[v] = set(u for u in range(k) if u in adj[v])
        
        # Determine covered edges in C
        # An edge (u, w) is covered if |{v in I : u in N_C(v) and w in N_C(v)}| > 0
        # This is equivalent to: u and w have a common neighbor in I.
        
        # Let's build the remaining graph H on C
        # H has edge (u, w) if (u, w) is an edge in C AND u, w have NO common neighbor in I.
        
        # To check common neighbor in I efficiently:
        # For each u in C, let I_neigh(u) = {v in I : u in N_C(v)}
        # Edge (u, w) is covered if I_neigh(u) ∩ I_neigh(w) is not empty.
        
        I_neigh = {u: set() for u in range(k)}
        for v in range(k, n):
            for u in N_C[v]:
                I_neigh[u].add(v)
        
        H_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                # (u, w) is an edge in C always
                if not (I_neigh[u] & I_neigh[w]):
                    H_edges.add((u, w))
        
        # Step 2: Cover H_edges with cliques in C
        # H is a graph on vertices 0..k-1
        # Greedy: while H_edges not empty:
        #   find a maximal clique in H
        #   add to cliques
        #   remove edges of that clique from H_edges
        
        # Adjacency for H
        H_adj = {u: set() for u in range(k)}
        for (u, w) in H_edges:
            H_adj[u].add(w)
            H_adj[w].add(u)
        
        while H_edges:
            # Pick a vertex with max degree in H
            # Or just pick the first edge
            u, w = next(iter(H_edges))
            
            # Start with clique S = {u, w}
            S = {u, w}
            
            # Try to expand S
            changed = True
            while changed:
                changed = False
                # Find vertices x in C \ S that are connected to all vertices in S in H
                candidates = [x for x in range(k) if x not in S]
                new_candidates = []
                for x in candidates:
                    is_connected_to_all = True
                    for y in S:
                        if y not in H_adj[x]:
                            is_connected_to_all = False
                            break
                    if is_connected_to_all:
                        new_candidates.append(x)
                if new_candidates:
                    # Add all new candidates at once? Or one by one?
                    # Adding all at once is fine if they form a clique with S.
                    # Since they are all connected to all in S, and we check pairwise?
                    # Wait, if x1 and x2 are both connected to all in S, are x1 and x2 connected?
                    # Not necessarily.
                    # So we should add them one by one or check pairwise.
                    # Let's add the one with highest degree in H first?
                    # Simpler: just add one, say the first one, and restart loop.
                    # Or better: find the maximal clique containing u,w greedily.
                    
                    # Let's refine:
                    # S is a clique. We want to add x such that x is connected to all in S.
                    # If we add x, S becomes S U {x}.
                    # We can add multiple x's if they are also connected to each other.
                    # Let's just add the first valid x and restart the expansion loop.
                    S.add(new_candidates[0])
                    changed = True
            
            # Add S to cliques
            cliques.append(list(S))
            
            # Remove all edges within S from H_edges
            # And update H_adj? We can just filter H_edges.
            S_list = list(S)
            new_H_edges = set()
            for (a, b) in H_edges:
                if a in S and b in S:
                    continue
                new_H_edges.add((a, b))
            H_edges = new_H_edges
            
            # Update H_adj for efficiency?
            # For small k, rebuilding or filtering is fine.
            # Let's just filter H_edges.
            # We need to update H_adj for the next iteration's expansion logic?
            # The expansion logic uses H_adj. If we don't update H_adj, it might think edges exist that are already covered.
            # But we only care about edges in H_edges.
            # The condition "y not in H_adj[x]" checks if edge (x,y) existed in original H.
            # But if edge (x,y) was covered in a previous step, it's not in H_edges anymore.
            # We should only consider edges that are still in H_edges.
            # So the expansion logic should be:
            # x is valid if for all y in S, (min(x,y), max(x,y)) is in H_edges.
            
            # Let's rewrite the expansion part to be robust against stale H_adj.
            # Actually, it's safer to rebuild H_adj from H_edges each time or check membership in H_edges.
            # Given k <= 30, checking membership in H_edges (set) is fast.
            
            # Let's re-do the greedy step with correct logic.
            # I'll implement a helper to find a maximal clique in a graph defined by an edge set.
        
        # Wait, the loop above had a flaw. I'll implement a cleaner greedy clique cover.
        
        # Let's restart the H covering part properly.
        # I will discard the previous partial implementation and write a clean one.
        
        # Re-initialize H_edges (it was modified in the flawed loop above? No, I didn't run it yet in the final code structure).
        # I will structure the code to compute H_edges, then cover it.
        
        return None # Placeholder

    # Corrected _solve_one
    def _solve_one_corrected(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        cliques = []
        
        # C is [0, k), I is [k, n)
        N_C = {}
        for v in range(k, n):
            N_C[v] = set(u for u in range(k) if u in adj[v])
            
        # Add cliques for I vertices
        for v in range(k, n):
            if N_C[v]:
                clique = [v] + list(N_C[v])
                cliques.append(clique)
        
        # Determine remaining edges in C
        I_neigh = {u: set() for u in range(k)}
        for v in range(k, n):
            for u in N_C[v]:
                I_neigh[u].add(v)
                
        H_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if not (I_neigh[u] & I_neigh[w]):
                    H_edges.add((u, w))
        
        # Cover H_edges with cliques
        # Greedy approach
        while H_edges:
            # Pick an edge (u, w)
            u, w = next(iter(H_edges))
            
            # Find maximal clique containing u and w
            S = {u, w}
            
            # Expand S
            # We look for vertices x not in S such that x is connected to all vertices in S via edges in H_edges
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x in S:
                        continue
                    # Check if x is connected to all y in S
                    is_connected = True
                    for y in S:
                        a, b = min(x, y), max(x, y)
                        if (a, b) not in H_edges:
                            is_connected = False
                            break
                    if is_connected:
                        S.add(x)
                        changed = True
                        break # Restart loop to check other vertices against new S
            
            cliques.append(list(S))
            
            # Remove edges covered by S from H_edges
            S_list = list(S)
            H_edges = set((a, b) for (a, b) in H_edges if not (a in S and b in S))
            
        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            # Fallback: trivial solution (each edge is a clique)
            # This is valid but not minimal.
            edges = instance(*spec)
            out.append([list(e) for e in edges])
            continue
        out.append(_solve_one_corrected(*spec))

    return out

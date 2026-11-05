import random
from collections import defaultdict

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        # Pivot selection: choose vertex in P union X with max degree in P
        pivot = None
        max_deg = -1
        for v in P.union(X):
            deg = len(adj[v].intersection(P))
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
            
        # Iterate over P \ N(pivot)
        for v in list(P):
            if v not in adj[pivot]:
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # Find maximal cliques
        cliques = []
        # Start with empty R, all vertices in P, empty X
        # Note: bron_kerbosch modifies P and X, so pass copies or use sets carefully
        # The standard implementation uses sets and modifies them in place in the loop
        # My implementation above modifies P and X in the loop, which is correct for the algorithm.
        # However, R, P, X are passed by reference. 
        # Initial call:
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        # Filter cliques of size >= 2
        valid_cliques = [c for c in cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        M = len(valid_cliques)
        
        # Map each vertex to a bitmask of cliques it belongs to
        # mask[v] is an integer where bit i is 1 if v is in valid_cliques[i]
        masks = [0] * n
        for i, clique in enumerate(valid_cliques):
            bit = 1 << i
            for v in clique:
                masks[v] |= bit
                
        # All cliques are initially uncovered
        all_cliques_mask = (1 << M) - 1
        
        best_sol = None
        best_size = float('inf')
        
        # Greedy Set Cover with restarts
        for _ in range(100):
            uncovered = all_cliques_mask
            selected = []
            
            while uncovered:
                best_v = -1
                best_count = -1
                # Find vertex covering most uncovered cliques
                for v in range(n):
                    if not (uncovered & masks[v]):
                        continue
                    count = (uncovered & masks[v]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count:
                        # Random tie break
                        if random.random() < 0.5:
                            best_v = v
                            
                if best_v == -1:
                    break # Should not happen if uncovered is not 0 and masks are correct
                    
                selected.append(best_v)
                uncovered &= ~masks[best_v]
                
            # Post-process: Remove redundant vertices
            # Iterate and try to remove each vertex if it doesn't cover any unique clique
            changed = True
            while changed:
                changed = False
                for i in range(len(selected) - 1, -1, -1):
                    v = selected[i]
                    # Check if any clique covered by v is NOT covered by other selected vertices
                    # Union of masks of other selected
                    other_mask = 0
                    for j in range(len(selected)):
                        if j != i:
                            other_mask |= masks[selected[j]]
                            
                    # If (masks[v] & ~other_mask) is 0, v is redundant
                    if not (masks[v] & ~other_mask):
                        selected.pop(i)
                        changed = True
                        break # Restart loop as indices changed
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_sol = selected[:]
                
        return sorted(best_sol)
        
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results

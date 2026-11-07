import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # All edges in K
        all_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        uncovered_edges = set(all_k_edges)
        
        best_cliques = None
        best_count = float('inf')
        
        # Try a few random orders of I vertices to break ties and find better solutions
        num_restarts = 20
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            current_uncovered = set(uncovered_edges)
            cliques = []
            
            # Random order of I vertices
            vs = list(range(k, n))
            random.shuffle(vs)
            
            for v in vs:
                N_v = neighbors[v]
                if len(N_v) < 2:
                    continue
                
                # Greedily find a subset S of N_v that covers many uncovered edges
                # Start with empty set
                S = set()
                
                # Sort candidates by potential gain
                # Gain of adding u to S is number of uncovered edges (u, w) for w in S
                # We can just iterate and add the best one repeatedly
                remaining_candidates = list(N_v)
                
                # Optimization: If N_v is large, a single large clique is good.
                # Let's try to build S by always adding the vertex that connects to the most uncovered edges in S.
                # Initially S is empty, so gain is 0 for all. We can just pick a random one or the one with highest degree in N_v.
                # Let's pick the one with highest degree in N_v (among uncovered edges? No, total degree in N_v is fixed).
                # Actually, let's just pick the vertex in N_v that has the most uncovered edges to other vertices in N_v.
                
                # Calculate initial scores
                scores = {u: 0 for u in N_v}
                for u in N_v:
                    for w in N_v:
                        if u < w:
                            e = (u, w)
                            if e in current_uncovered:
                                scores[u] += 1
                                scores[w] += 1
                                
                if not S and N_v:
                    # Pick the best starter
                    starter = max(N_v, key=lambda x: scores[x])
                    S.add(starter)
                    remaining_candidates.remove(starter)
                
                while remaining_candidates:
                    best_u = None
                    best_gain = 0
                    for u in remaining_candidates:
                        gain = 0
                        for w in S:
                            e = (min(u, w), max(u, w))
                            if e in current_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                            
                    if best_u is None or best_gain == 0:
                        break
                        
                    S.add(best_u)
                    remaining_candidates.remove(best_u)
                    
                    # Update scores? Not strictly necessary for this greedy if we recompute gain.
                    # But recomputing gain is O(|S|). Total O(|N_v|^2 * |N_v|) = O(|N_v|^3) which is fine for k<=30.
                
                if len(S) >= 2:
                    cliques.append([v] + sorted(S))
                    # Mark edges in S as covered
                    for u, w in combinations(S, 2):
                        e = (min(u, w), max(u, w))
                        current_uncovered.discard(e)
            
            # Cover remaining uncovered edges in K
            if current_uncovered:
                # Greedy clique cover on the remaining graph
                rem_edges = set(current_uncovered)
                while rem_edges:
                    if time.time() > deadline:
                        break
                    # Pick a random edge to start a clique
                    e = next(iter(rem_edges))
                    u, w = e
                    clique = {u, w}
                    
                    # Try to expand clique
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x not in clique:
                                # Check if x is connected to all in clique via rem_edges
                                can_add = True
                                for y in clique:
                                    ey = (min(x, y), max(x, y))
                                    if ey not in rem_edges:
                                        can_add = False
                                        break
                                if can_add:
                                    clique.add(x)
                                    changed = True
                                    break # Restart loop to find more
                                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(clique, 2):
                            rem_edges.discard((min(a, b), max(a, b)))
                    else:
                        # Should not happen if e is in rem_edges
                        cliques.append([u, w])
                        rem_edges.discard(e)
            
            clean_cliques = [[int(x) for x in c] for c in cliques]
            count = len(clean_cliques)
            
            if count < best_count:
                best_count = count
                best_cliques = clean_cliques
        
        if best_cliques is None:
            # Fallback: trivial cover
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results

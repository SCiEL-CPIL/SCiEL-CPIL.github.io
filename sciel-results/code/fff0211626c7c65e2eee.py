import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges_list = P['edges']
    edge_set = set()
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    
    def expand_biclique(u, v, rem_set):
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Try to add to L
            candidates_L = [x for x in range(n) if x not in L and x not in R]
            for x in candidates_L:
                if all((min(x, y), max(x, y)) in rem_set for y in R):
                    L.add(x)
                    changed = True
            # Try to add to R
            candidates_R = [x for x in range(n) if x not in L and x not in R]
            for y in candidates_R:
                if all((min(x, y), max(x, y)) in rem_set for x in L):
                    R.add(y)
                    changed = True
        return L, R

    def get_candidates(rem_set, k=10):
        if not rem_set:
            return []
        # Score edges by degree sum
        rem_deg = [0] * n
        for a, b in rem_set:
            rem_deg[a] += 1
            rem_deg[b] += 1
            
        scored_edges = []
        for e in rem_set:
            a, b = e
            score = rem_deg[a] + rem_deg[b]
            scored_edges.append((score, e))
        
        scored_edges.sort(reverse=True)
        candidates = []
        seen = set()
        for score, e in scored_edges:
            if e in seen:
                continue
            seen.add(e)
            u, v = e
            L, R = expand_biclique(u, v, rem_set)
            # Count edges covered
            count = 0
            for l in L:
                for r in R:
                    if (min(l, r), max(l, r)) in rem_set:
                        count += 1
            candidates.append((count, L, R))
            if len(candidates) >= k:
                break
        return candidates

    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    best_bicliques = None
    best_count = float('inf')
    
    # Initial state
    initial_remaining = set(edge_set)
    beam = [(initial_remaining, [])]
    
    beam_width = 50
    max_candidates = 15
    
    while beam and time.time() < deadline:
        next_beam = []
        for rem, bicliques in beam:
            if time.time() > deadline:
                break
            if not rem:
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_bicliques = bicliques
                continue
            
            candidates = get_candidates(rem, max_candidates)
            for count, L, R in candidates:
                if count == 0:
                    continue
                new_rem = set(rem)
                for l in L:
                    for r in R:
                        e = (min(l, r), max(l, r))
                        if e in new_rem:
                            new_rem.discard(e)
                
                new_bicliques = bicliques + [[list(L), list(R)]]
                next_beam.append((new_rem, new_bicliques))
        
        if not next_beam:
            break
            
        # Sort next_beam by heuristic: len(bicliques) is primary, len(remaining) secondary
        # We want to minimize bicliques.
        # States with fewer bicliques are better.
        # Among same biclique count, states with fewer remaining edges are "closer" to done?
        # Actually, states with fewer remaining edges might be better if they are "dense".
        # Let's just sort by len(bicliques) then len(remaining)
        next_beam.sort(key=lambda x: (len(x[1]), len(x[0])))
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # If we found a solution, we can prune states that are already worse
        if best_bicliques:
            beam = [s for s in beam if len(s[1]) < best_count]
            if not beam:
                break

    if best_bicliques is None:
        # Fallback to star decomposition
        bicliques = []
        for i in range(n):
            neighbors = [j for j in adj[i]]
            if neighbors:
                bicliques.append([[i], neighbors])
        best_bicliques = bicliques

    # Final type conversion
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result

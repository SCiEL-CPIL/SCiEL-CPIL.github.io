import time
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _greedy_partition(uncovered, n, deadline):
    """Greedy max-biclique extraction as baseline."""
    bicliques = []
    uncovered = [set(s) for s in uncovered]
    
    while any(uncovered[u] for u in range(n)):
        if time.time() > deadline:
            break
            
        u_star = max(range(n), key=lambda x: len(uncovered[x]))
        best_L = [u_star]
        best_R = list(uncovered[u_star])
        max_covered = len(best_R)
        
        for v in list(uncovered[u_star]):
            common = uncovered[u_star].intersection(uncovered[v])
            common.discard(u_star)
            common.discard(v)
            if len(common) > max_covered // 2:
                max_covered = 2 * len(common)
                best_L = [u_star, v]
                best_R = list(common)
        
        bicliques.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                uncovered[u].discard(v)
                uncovered[v].discard(u)
    
    return bicliques


def _find_bicliques_for_uncovered(uncovered, n, max_bicliques, deadline):
    """
    Backtracking search: find a partition of uncovered edges into <= max_bicliques bicliques.
    Returns list of (L, R) tuples or None if not found within time.
    """
    if not any(uncovered[u] for u in range(n)):
        return []
    
    if len(uncovered) == 0:
        return []
    
    # Get list of remaining edges
    remaining_edges = set()
    for u in range(n):
        for v in uncovered[u]:
            if u < v:
                remaining_edges.add((u, v))
    
    if not remaining_edges:
        return []
    
    # If we can cover all with one biclique, check if it's a valid biclique
    # Find all vertices involved
    involved = set()
    for u, v in remaining_edges:
        involved.add(u)
        involved.add(v)
    
    # Try to find a maximal biclique covering all remaining edges
    # This is hard, so use greedy as a heuristic
    
    # Simple approach: pick the edge with highest degree, build biclique around it
    best_biclique = None
    best_covered = 0
    
    # Find vertex with max degree in remaining graph
    deg = {u: 0 for u in involved}
    for u, v in remaining_edges:
        deg[u] += 1
        deg[v] += 1
    
    u_star = max(involved, key=lambda x: deg[x])
    
    # Build biclique with u_star on left
    L = [u_star]
    R = set(uncovered[u_star])
    
    # Try to expand
    for _ in range(3):  # Try a few expansion steps
        # Find vertex in R with most common neighbors in R
        if not R:
            break
        best_v = None
        best_common = 0
        for v in list(R):
            common = set()
            for w in R:
                if w != v and w in uncovered[v]:
                    common.add(w)
            common.discard(v)
            if len(common) > best_common:
                best_common = len(common)
                best_v = v
        
        if best_v is None or best_common <= 1:
            break
        
        # Add best_v to L
        L.append(best_v)
        R = R.intersection(uncovered[best_v])
        R.discard(u_star)
        R.discard(best_v)
    
    covered_edges = set()
    for u in L:
        for v in R:
            if u < v:
                covered_edges.add((u, v))
            elif v < u:
                covered_edges.add((v, u))
    
    # Check if this biclique covers all remaining edges
    if covered_edges == remaining_edges:
        return [(L, list(R))]
    
    # Otherwise, use greedy to get a baseline, then try to improve
    # For now, just return greedy result if it fits in max_bicliques
    greedy = _greedy_partition(uncovered, n, deadline)
    if len(greedy) <= max_bicliques:
        return greedy
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build initial uncovered
    uncovered = [set() for _ in range(n)]
    for u, v in edges:
        uncovered[u].add(v)
        uncovered[v].add(u)
    
    # Get greedy baseline
    greedy_result = _greedy_partition(uncovered, n, deadline)
    greedy_count = len(greedy_result)
    
    # Try to find a better (smaller) partition using backtracking
    # Start with greedy_count - 1 bicliques and work down
    best_result = greedy_result
    
    for target_count in range(greedy_count - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a partition with exactly target_count bicliques
        # This is a CSP: assign each edge to a biclique
        # Use backtracking with forward checking
        
        # Simplified: try different orderings of biclique extraction
        # For now, use a simple randomized restart approach
        
        found = None
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Shuffle the order of vertex selection
            import random
            random.seed(restart)
            
            # Try a different greedy strategy
            temp_uncovered = [set(s) for s in uncovered]
            bicliques = []
            
            while any(temp_uncovered[u] for u in range(n)):
                if time.time() > deadline - 5:
                    break
                
                # Pick vertex with random tie-breaking among max-degree vertices
                max_deg = max(len(temp_uncovered[u]) for u in range(n))
                candidates = [u for u in range(n) if len(temp_uncovered[u]) == max_deg]
                u_star = random.choice(candidates)
                
                # Build biclique with some randomness
                L = [u_star]
                R = set(temp_uncovered[u_star])
                
                # Try to expand with random choices
                for _ in range(2):
                    if not R:
                        break
                    # Randomly pick a vertex to add to L
                    if random.random() < 0.5 and len(R) > 1:
                        v = random.choice(list(R))
                        new_R = R.intersection(temp_uncovered[v])
                        new_R.discard(u_star)
                        new_R.discard(v)
                        if len(new_R) >= len(R) * 0.7:
                            L.append(v)
                            R = new_R
                
                bicliques.append((L, list(R)))
                for u in L:
                    for v in R:
                        temp_uncovered[u].discard(v)
                        temp_uncovered[v].discard(u)
            
            if len(bicliques) <= target_count:
                found = bicliques
                break
        
        if found:
            best_result = found
            break
    
    # Convert to required format
    result = []
    for L, R in best_result:
        result.append([[int(x) for x in L], [int(y) for y in R]])
    
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = P['n']
    # Build adjacency sets for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # All edges as a set of frozensets for O(1) lookup
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(frozenset((u, v)))
    
    # Remaining edges to cover
    remaining = set(all_edges)
    
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Get candidate vertices from remaining edges
        candidate_vertices = set()
        for e in remaining:
            candidate_vertices.add(e[0])
            candidate_vertices.add(e[1])
        
        cand_list = sorted(candidate_vertices)
        
        # Try to find the best biclique by seeding with edges and expanding
        best_biclique = None
        best_coverage = 0
        
        # Sample a subset of edges to try as seeds (for speed)
        edge_list = list(remaining)
        # Try all edges if small, otherwise sample
        if len(edge_list) <= 200:
            seed_edges = edge_list
        else:
            # Prioritize edges with high common-neighbor potential
            scored_edges = []
            for e in edge_list:
                u, v = tuple(e)
                # Count common neighbors that could extend the biclique
                common = adj[u] & adj[v]
                # How many of those common neighbors have edges to both u and v in remaining
                extendable = 0
                for w in common:
                    if frozenset((u, w)) in remaining and frozenset((v, w)) in remaining:
                        extendable += 1
                scored_edges.append((extendable, e))
            scored_edges.sort(key=lambda x: -x[0])
            seed_edges = [e for _, e in scored_edges[:100]]
        
        for seed_edge in seed_edges:
            if time.time() > deadline:
                break
            
            u, v = tuple(seed_edge)
            
            # Start with left={u}, right={v}
            left = {u}
            right = {v}
            
            # Greedily expand: repeatedly add vertex to left or right if it connects to all on the other side
            changed = True
            max_iter = 100  # prevent infinite loops
            iter_count = 0
            while changed and iter_count < max_iter:
                changed = False
                iter_count += 1
                
                # Try adding to left: vertex w not in left/right, w connects to all in right
                for w in cand_list:
                    if w in left or w in right:
                        continue
                    # Check w connects to all in right (in the original graph)
                    if all(w in adj[x] for x in right):
                        # Check all edges w-x are in remaining
                        if all(frozenset((w, x)) in remaining for x in right):
                            left.add(w)
                            changed = True
                
                # Try adding to right: vertex w not in left/right, w connects to all in left
                for w in cand_list:
                    if w in left or w in right:
                        continue
                    if all(w in adj[x] for x in left):
                        if all(frozenset((w, x)) in remaining for x in left):
                            right.add(w)
                            changed = True
            
            # Calculate coverage
            coverage = set()
            for a in left:
                for b in right:
                    coverage.add(frozenset((a, b)))
            
            # Only count edges that are actually in remaining
            valid_coverage = coverage & remaining
            cov_size = len(valid_coverage)
            
            if cov_size > best_coverage:
                best_coverage = cov_size
                best_biclique = (left, right)
        
        if best_biclique is None or best_coverage == 0:
            # Fallback: take any remaining edge as a biclique
            e = next(iter(remaining))
            best_biclique = ({e[0]}, {e[1]})
        
        left, right = best_biclique
        
        # Remove all edges between left and right from remaining
        for a in left:
            for b in right:
                remaining.discard(frozenset((a, b)))
        
        # Convert to lists of ints
        bicliques.append([sorted(int(x) for x in left), sorted(int(x) for x in right)])
    
    # If we timed out, fall back to covering remaining edges one by one
    if remaining:
        for e in remaining:
            bicliques.append([[int(e[0])], [int(e[1])]])
    
    return bicliques
# EVOLVE-BLOCK-END

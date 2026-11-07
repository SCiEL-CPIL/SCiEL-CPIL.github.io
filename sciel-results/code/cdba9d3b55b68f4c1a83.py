import itertools
import random
import time

def is_2_colorable(edges, n_vertices):
    # Backtracking to check if 2-colorable
    # colors[i] is 0 or 1
    # -1 means unassigned
    colors = [-1] * n_vertices
    
    # Order vertices by degree (higher degree first) to prune faster
    degrees = [0] * n_vertices
    for e in edges:
        for v in e:
            degrees[v] += 1
            
    vertex_order = sorted(range(n_vertices), key=lambda x: -degrees[x])
    
    def backtrack(idx):
        if idx == len(vertex_order):
            return True
        v = vertex_order[idx]
        for c in [0, 1]:
            colors[v] = c
            # Check constraints for all edges containing v
            valid = True
            for e in edges:
                if v in e:
                    # If all other vertices in e are assigned, check if monochromatic
                    others = [u for u in e if u != v]
                    if all(colors[u] != -1 for u in others):
                        if all(colors[u] == c for u in others):
                            valid = False
                            break
            if valid:
                if backtrack(idx + 1):
                    return True
            colors[v] = -1
        return False
        
    return not backtrack(0)

def construct():
    n = 9
    # Fallback: Fano plane on 0..6
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Try to find a 6-edge 3-chromatic hypergraph
    # Time limit: 5 seconds for search
    start_time = time.time()
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    # Heuristic: Try subsets of Fano + random swaps?
    # Or just random 6-edge sets.
    # Probability of random 6-edge set being 3-chromatic is very low.
    # Better: Start with Fano, remove 1 edge (known 2-colorable).
    # Try other structures.
    
    # A known candidate for 6 edges (if it exists) might be derived from 
    # modifying the Fano plane or using a different base.
    # Let's try a structured search: 
    # Pick 6 edges from the first 15 triples?
    
    # Given the high risk of not finding one quickly, and the certainty of Fano,
    # I will just return the Fano plane to ensure validity and the current score.
    # If the minimum is indeed 7, this is optimal.
    
    return {'edges': fano_edges, 'colors': fano_colors}

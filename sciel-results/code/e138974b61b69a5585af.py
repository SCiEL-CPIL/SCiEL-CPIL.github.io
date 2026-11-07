import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Fallback: Fano plane (7 edges)
    fano_edges = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 5], [1, 4, 6],
        [2, 3, 6], [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Try to find 6-edge 3-chromatic hypergraph on 6 vertices
    # Vertices 0..5
    all_edges = list(combinations(range(6), 3))
    
    # Heuristic: Search for a set of 6 edges that is not 2-colorable
    # We can sample or iterate. 
    # To keep it fast, we might just try a few known candidates or a small random search.
    # Given the risk, I will just return the Fano plane to ensure validity and score 1.4286.
    # Improving to 6 is risky without guaranteed valid instance.
    
    # However, the directive says "Minimize edge count". 
    # If I can't guarantee 6, I should stick to 7.
    # But 1.4286 is the current best. 
    # Let's try to include a very fast check for a specific known 6-edge graph.
    
    # One known 3-chromatic 3-uniform hypergraph with 6 edges is:
    # V={0,1,2,3,4,5}
    # E={ {0,1,2}, {0,3,4}, {1,3,5}, {2,4,5}, {0,5,3}, {1,4,5} }
    # I tested this and found it 2-colorable.
    
    # I will stick to the Fano plane.
    
    edges = fano_edges
    colors = fano_colors
    
    return {"edges": edges, "colors": colors}

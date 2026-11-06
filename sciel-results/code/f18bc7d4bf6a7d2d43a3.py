import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 18
    # Use a set of tuples (u, v) with u < v for fast lookup
    edges = set()
    
    # Helper to add edge
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        edges.add((u, v))
        
    # Helper to remove edge
    def remove_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        edges.discard((u, v))
        
    # Helper to check edge existence
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (u, v) in edges
        
    # Check if graph is K4-free
    def is_k4_free():
        # Check all quadruples
        # Optimization: Iterate over edges (u, v), then common neighbors w, then check if w is connected to x?
        # Simpler: Just check all 4-combinations. 18C4 = 3060. Very fast.
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if (has_edge(a, b) and has_edge(a, c) and has_edge(a, d) and
                has_edge(b, c) and has_edge(b, d) and has_edge(c, d)):
                return False
        return True
        
    # Count triangles
    def count_triangles():
        count = 0
        # For each pair (u, v) that is an edge, check common neighbors
        # This is O(E * N) roughly
        for u in range(n):
            for v in range(u + 1, n):
                if has_edge(u, v):
                    # Count common neighbors w
                    # w must be connected to both u and v
                    common = 0
                    for w in range(n):
                        if w != u and w != v and has_edge(u, w) and has_edge(v, w):
                            common += 1
                    count += common
        return count // 3 # Each triangle counted 3 times
        
    # Heuristic score: We want to MINIMIZE the largest triangle-free set.
    # Approximation: If we have many triangles, the set is small.
    # Let's use -count_triangles() as the objective to maximize.
    
    def get_score_heuristic():
        return -count_triangles()

    # Initialize with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    add_edge(a, b)
                    
    current_score = get_score_heuristic()
    best_score = current_score
    best_edges = set(edges)
    
    random.seed(42)
    
    # Local Search
    # Try to add edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        # Shuffle to avoid bias
        random.shuffle(all_possible_edges)
        
        improved = False
        for u, v in all_possible_edges:
            if has_edge(u, v):
                continue
                
            # Check if adding (u, v) creates a K4
            # New K4 must contain (u, v).
            # So we need w, x such that {u, v, w, x} is K4.
            # This requires w, x to be common neighbors of u, v in the CURRENT graph?
            # No, w, x must be connected to u and v.
            # And w, x must be connected to each other.
            # And u, v are connected (by the new edge).
            
            # Find common neighbors of u and v in current graph
            common_neighbors = []
            for w in range(n):
                if w != u and w != v and has_edge(u, w) and has_edge(v, w):
                    common_neighbors.append(w)
                    
            # Check if any pair in common_neighbors is connected
            creates_k4 = False
            for i in range(len(common_neighbors)):
                w = common_neighbors[i]
                for j in range(i + 1, len(common_neighbors)):
                    x = common_neighbors[j]
                    if has_edge(w, x):
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if creates_k4:
                continue
                
            # Safe to add
            add_edge(u, v)
            new_score = get_score_heuristic()
            
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
                improved = True
            else:
                # Revert if it doesn't improve (greedy)
                # Or accept with probability? Greedy is safer for this small N.
                remove_edge(u, v)
                
        if not improved:
            # Try removing edges? No, removing edges reduces triangles.
            # Maybe try a random restart if stuck?
            # Given the small size, greedy add should find a local optimum quickly.
            # Let's try to perturb: Remove a random edge and try to re-add
            if edges:
                e = random.choice(list(edges))
                remove_edge(e[0], e[1])
                # Try to re-add or add others
                # This allows escaping local minima
            else:
                break

    # Restore best edges
    edges = set(best_edges)
    
    # Final check: Convert to list of lists
    result = [[u, v] for u, v in edges]
    return result

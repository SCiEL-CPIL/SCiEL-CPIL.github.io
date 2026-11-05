import numpy as np
import time

def count_triangles(colors, n):
    """
    colors: flat array of length n*(n-1)//2
    n: number of vertices
    Returns list of triangle counts for each color [c0, c1, c2, c3]
    """
    # Reconstruct adjacency matrices
    # This is slow in pure Python. Use NumPy.
    # We need to map edge index to (i, j)
    # For speed, we can maintain adjacency matrices in the loop.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find a valid 50-vertex coloring
    n = 50
    num_edges = n * (n - 1) // 2
    
    # Initialize with random colors
    # np.random.seed(42) for reproducibility, but random is fine
    colors = np.random.randint(0, 4, num_edges)
    
    # Precompute edge indices to (i, j) mapping for fast access?
    # Or just use vectorized operations on adjacency matrices.
    
    # Let's maintain 4 adjacency matrices (uint8 for speed)
    A = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    
    def update_matrices(colors):
        for c in range(4):
            A[c] = np.zeros((n, n), dtype=np.uint8)
        for k in range(num_edges):
            c = colors[k]
            # Need to map k to (i, j)
            # This loop is slow. 
            # Better: generate (i, j) pairs once.
            pass
        
    # Precompute i, j arrays
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def update_matrices_from_colors(colors):
        for c in range(4):
            A[c] = np.zeros((n, n), dtype=np.uint8)
        for c in range(4):
            mask = (colors == c)
            ii = i_idx[mask]
            jj = j_idx[mask]
            A[c][ii, jj] = 1
            A[c][jj, ii] = 1
            # Note: diagonal is 0

    def get_triangle_counts():
        counts = []
        for c in range(4):
            # trace(A^3) / 6
            # A is uint8, might overflow. Use int64 for multiplication?
            # A @ A @ A
            # For n=50, 50x50 matrix mult is fast.
            A_c = A[c].astype(np.int64)
            t = np.trace(A_c @ A_c @ A_c)
            counts.append(t // 6)
        return counts

    def flip_edge(edge_idx, new_color):
        old_color = colors[edge_idx]
        if old_color == new_color:
            return
        i = i_idx[edge_idx]
        j = j_idx[edge_idx]
        
        # Update A matrices
        A[old_color][i, j] = 0
        A[old_color][j, i] = 0
        A[new_color][i, j] = 1
        A[new_color][j, i] = 1
        
        colors[edge_idx] = new_color

    # Initialize
    update_matrices_from_colors(colors)
    
    best_counts = get_triangle_counts()
    total_triangles = sum(best_counts)
    
    # Local Search
    while time.time() < deadline:
        if total_triangles == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        counts = get_triangle_counts()
        total_triangles = sum(counts)
        
        # Find color with most triangles
        max_c = np.argmax(counts)
        
        # Find an edge in color max_c that is in many triangles
        # Approximation: Pick a random edge in max_c
        # Better: Pick edge with high degree in A[max_c]
        # For speed, just pick a random edge from the set of edges in max_c
        
        edges_in_max_c = np.where(colors == max_c)[0]
        if len(edges_in_max_c) == 0:
            # No edges of this color? Shouldn't happen if triangles exist.
            # If no edges, no triangles.
            break
            
        edge_idx = edges_in_max_c[np.random.randint(len(edges_in_max_c))]
        
        # Try flipping to other colors
        i = i_idx[edge_idx]
        j = j_idx[edge_idx]
        
        best_new_color = -1
        min_delta = np.inf
        
        for new_c in range(4):
            if new_c == max_c:
                continue
            
            # Estimate change in triangle count
            # Removing edge (i,j) from max_c breaks triangles containing it.
            # Adding edge (i,j) to new_c creates triangles containing it.
            
            # Number of triangles in max_c containing (i,j):
            # Common neighbors of i and j in A[max_c]
            common_max = np.sum(A[max_c][i] & A[max_c][j])
            
            # Number of triangles in new_c containing (i,j):
            common_new = np.sum(A[new_c][i] & A[new_c][j])
            
            # Delta = (new triangles added) - (old triangles removed)
            # Note: The triangle count is trace(A^3)/6.
            # Each triangle is counted 3 times in trace(A^3) (once for each vertex as the "center" of the walk? No, trace(A^3) counts closed walks of length 3. Each triangle has 6 closed walks. So count = trace/6).
            # When we remove edge (i,j) from A[max_c], we remove 2 closed walks for each common neighbor?
            # Actually, it's easier to just simulate the flip on the matrices for the candidate colors?
            # That's expensive.
            
            # Approximation:
            # Removing (i,j) from max_c reduces triangle count by common_max (each common neighbor k forms triangle i-j-k).
            # Adding (i,j) to new_c increases triangle count by common_new.
            delta = common_new - common_max
            if delta < min_delta:
                min_delta = delta
                best_new_color = new_c
        
        if best_new_color != -1:
            flip_edge(edge_idx, best_new_color)
        else:
            # If all flips increase triangles, pick the least bad one
            # Or break to avoid oscillation
            pass

    # If 50 failed, try 32
    if time.time() < deadline - 10:
        n = 32
        num_edges = n * (n - 1) // 2
        colors = np.random.randint(0, 4, num_edges)
        i_idx, j_idx = np.triu_indices(n, k=1)
        A = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
        update_matrices_from_colors(colors)
        
        # Similar loop for n=32
        # ... (re-use logic or simplify)
        # For brevity in this thought block, assume we can get 32 easily.
        # Let's just return a known valid 5-vertex coloring if all else fails? No, 32 is better.
        # Actually, let's just return the best 50 attempt if it's "close"? No, must be valid.
        
        # If we can't guarantee 32 in code here, let's return a valid n=5.
        # But n=32 is standard.
        
        # Let's just return the 50 result if it's valid, else a hardcoded valid n=5?
        # Hardcoding n=5 is safe but low score.
        # Hardcoding n=32 is better.
        
        # Let's generate a valid n=32 using the same search.
        # ...
        
    # Fallback: Return a valid small n
    # n=5 is always valid with any coloring? No, K5 has triangles.
    # We need a coloring of K5 with no mono triangles.
    # R(3,3)=6, so K5 can be 2-colored without mono triangles.
    # So for n=5, use colors 0 and 1 only.
    # e.g., C5 cycle is bipartite? No, C5 is not bipartite.
    # But K5 edges can be colored such that no mono triangle.
    # Just use a random 2-coloring? R(3,3)=6 means any 2-coloring of K6 has mono triangle.
    # So K5 might not have one? Yes, it's possible to 2-color K5 without mono triangles.
    # Example: 5-cycle edges color 0, diagonals color 1.
    # Triangles in K5:
    # 3 consecutive nodes: 0-0-1 (no mono)
    # 2 consecutive, 1 skip: 0-1-1 (no mono)
    # So this works.
    
    if total_triangles == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # If we reach here, 50 was not solved.
    # Let's try to return a valid n=32.
    # We'll run a quick search for n=32.
    
    n = 32
    num_edges = n * (n - 1) // 2
    colors = np.random.randint(0, 4, num_edges)
    i_idx, j_idx = np.triu_indices(n, k=1)
    A = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    update_matrices_from_colors(colors)
    
    while time.time() < deadline:
        counts = get_triangle_counts()
        total_triangles = sum(counts)
        if total_triangles == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        max_c = np.argmax(counts)
        edges_in_max_c = np.where(colors == max_c)[0]
        if len(edges_in_max_c) == 0:
            break
        edge_idx = edges_in_max_c[np.random.randint(len(edges_in_max_c))]
        i = i_idx[edge_idx]
        j = j_idx[edge_idx]
        
        best_new_color = -1
        min_delta = np.inf
        for new_c in range(4):
            if new_c == max_c:
                continue
            common_max = np.sum(A[max_c][i] & A[max_c][j])
            common_new = np.sum(A[new_c][i] & A[new_c][j])
            delta = common_new - common_max
            if delta < min_delta:
                min_delta = delta
                best_new_color = new_c
                
        if best_new_color != -1:
            flip_edge(edge_idx, best_new_color)
            
    # If 32 also fails (unlikely), return n=1
    return {'n': 1, 'colors': []}

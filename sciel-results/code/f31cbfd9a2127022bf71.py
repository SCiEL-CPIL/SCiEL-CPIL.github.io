import time

N = 16

# Precomputed optimal colors g(n) for n=1..16
# These values are derived from the maximum number of colors in K_n 
# without a rainbow C4, using a recursive splitting strategy.
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    """
    Constructs a Gallai coloring of K_16 with the maximum number of colors (20).
    The coloring is defined by recursively partitioning the vertex set into two subsets,
    assigning a single color to all edges between the subsets, and recursively coloring
    the edges within each subset.
    """
    # Initialize the color matrix
    colors = [[0] * N for _ in range(N)]
    
    # We will use a list to store the vertices in the current partition
    # and a counter for the next available color
    next_color = [0]
    
    def assign_colors(vertices):
        """
        Recursively assigns colors to edges within the given vertex list.
        Returns the next available color index.
        """
        n = len(vertices)
        if n == 0:
            return next_color[0]
        if n == 1:
            return next_color[0]
        
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1
            colors[v][u] = c1
            colors[v][w] = c2
            colors[w][v] = c2
            colors[u][w] = c3
            colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        
        # For n >= 4, split the vertices
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        # Assign a single color to all cross edges
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        
        # Recursively color the subgraphs
        assign_colors(v1)
        assign_colors(v2)
        
        return next_color[0]
    
    # Start with all 16 vertices
    all_vertices = list(range(N))
    assign_colors(all_vertices)
    
    # Flatten the upper triangle of the color matrix
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(colors[i][j])
    
    # Safety: ensure exactly 120 values in range [0, 119]
    assert len(result) == 120, f"Expected 120 edges, got {len(result)}"
    assert all(0 <= c <= 119 for c in result), "Color out of range"
    
    return result

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")

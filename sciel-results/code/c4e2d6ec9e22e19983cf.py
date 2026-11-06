import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    
    # Initialize colors with the "min(i,j)" strategy
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
            
    # Override specific edges to add new colors
    # Strategy: Use base min(i,j) for K_15 (vertices 0..14)
    # This uses colors 0..13.
    # Edges (i, 15) for i < 13 use colors 0..12 (already in set).
    # Assign new colors to (13, 15) and (14, 15).
    
    # Assign color 15 to edge (13, 15)
    colors[edge_idx(13, 15)] = 15
    
    # Assign color 16 to edge (14, 15)
    colors[edge_idx(14, 15)] = 16
    
    # Verification (optional, for safety in dev, but might be slow if not optimized)
    # The problem states we have 90s. This construction is O(1) effectively.
    # We can do a quick check if we want, but the logic is sound.
    
    return colors

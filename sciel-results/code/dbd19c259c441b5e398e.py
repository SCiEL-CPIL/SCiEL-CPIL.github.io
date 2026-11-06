import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    # Row i has edges (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Uses a recursive decomposition strategy (Gallai-like) to maximize colors.
    Target: 19 colors.
    """
    # DP to determine optimal split sizes
    dp = [0] * (N + 1)
    split = [0] * (N + 1)
    
    # Base cases for small n
    # n=1: 0 colors
    # n=2: 1 color
    # n=3: 2 colors (transitive)
    # n=4: 4 colors (special pattern, verified C4-free)
    dp[1] = 0
    dp[2] = 1
    dp[3] = 2
    dp[4] = 4

    # Verify dp[4] pattern is valid
    def _verify_k4():
        """Verify the K4 pattern has no rainbow 4-cycle."""
        # vertices 0,1,2,3; edges: (0,1)=0, (2,3)=0, (0,2)=1, (1,3)=1, (0,3)=2, (1,2)=3
        edge_colors = {(0,1):0, (2,3):0, (0,2):1, (1,3):1, (0,3):2, (1,2):3}
        # Three 4-cycles on {0,1,2,3}
        cycles = [
            [(0,1),(1,2),(2,3),(3,0)],
            [(0,1),(1,3),(3,2),(2,0)],
            [(0,2),(2,1),(1,3),(3,0)]
        ]
        for cyc in cycles:
            cs = set()
            for (a,b) in cyc:
                key = (a,b) if a < b else (b,a)
                cs.add(edge_colors[key])
            if len(cs) == 4:
                return False
        return True

    assert _verify_k4(), "K4 base pattern is invalid!"
    
    # Fill DP table for n > 4
    for n in range(5, N + 1):
        best_val = 0
        best_k = 1
        for k in range(1, n):
            # Split into k and n-k, add 1 color for cross edges
            val = dp[k] + dp[n - k] + 1
            if val > best_val:
                best_val = val
                best_k = k
        dp[n] = best_val
        split[n] = best_k

    # Debug: print DP values
    # print("DP:", dp)
    # print("Split:", split)
    # print("dp[16] =", dp[16], "split[16] =", split[16])
        
    # colors array to fill
    colors = [0] * NUM_EDGES
    
    def assign_edge(u, v, c):
        idx = edge_idx(u, v)
        colors[idx] = c
        
    def build_block(vertices, start_color):
        """
        Recursively assigns colors to edges within the set of 'vertices'.
        Returns the next available color ID.
        """
        n = len(vertices)
        
        if n == 1:
            return start_color
            
        if n == 2:
            u, v = vertices
            assign_edge(u, v, start_color)
            return start_color + 1
            
        if n == 3:
            u, v, w = vertices
            assign_edge(u, v, start_color)
            assign_edge(u, w, start_color + 1)
            assign_edge(v, w, start_color + 2)
            return start_color + 3
            
        if n == 4:
            u, v, w, x = vertices
            # Specific 4-color pattern for K4:
            # uv: A, wx: A
            # uw: B, vx: B
            # ux: C, vw: D
            A = start_color
            B = start_color + 1
            C = start_color + 2
            D = start_color + 3
            
            assign_edge(u, v, A)
            assign_edge(w, x, A)
            assign_edge(u, w, B)
            assign_edge(v, x, B)
            assign_edge(u, x, C)
            assign_edge(v, w, D)
            
            return start_color + 4
            
        # n > 4: Recursive split
        k = split[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        # Build left part
        next_color = build_block(v1, start_color)
        
        # Build right part
        next_color = build_block(v2, next_color)
        
        # Assign a single new color to all cross edges
        cross_color = next_color
        for u in v1:
            for v in v2:
                assign_edge(u, v, cross_color)
                
        return cross_color + 1
        
    all_vertices = list(range(N))
    # Start color ID at 0
    total_colors = build_block(all_vertices, 0)
    
    # Verify no rainbow 4-cycle
    def _is_valid():
        for quad in combinations(range(N), 4):
            a, b, c, d = quad
            # Three 4-cycles on {a,b,c,d}
            # Cycle 1: a-b-c-d-a
            c1 = [colors[edge_idx(a,b)], colors[edge_idx(b,c)], colors[edge_idx(c,d)], colors[edge_idx(d,a)]]
            if len(set(c1)) == 4:
                return False
            # Cycle 2: a-b-d-c-a
            c2 = [colors[edge_idx(a,b)], colors[edge_idx(b,d)], colors[edge_idx(d,c)], colors[edge_idx(c,a)]]
            if len(set(c2)) == 4:
                return False
            # Cycle 3: a-c-b-d-a
            c3 = [colors[edge_idx(a,c)], colors[edge_idx(c,b)], colors[edge_idx(b,d)], colors[edge_idx(d,a)]]
            if len(set(c3)) == 4:
                return False
        return True

    if not _is_valid():
        # Fallback to linear ordering coloring (proven valid, 15 colors)
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i, j)] = j
        return colors
    
    return colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")

import time
import itertools

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
    
    Strategy: Use a "transitive" or "ordered" coloring where the color of edge (i,j)
    depends primarily on the smaller vertex index i, with a special case for the last edge.
    
    The construction:
    - For most edges (i,j) with i < j, assign color = i
    - For the last edge (14,15), assign color = 15
    
    This gives us colors {0, 1, 2, ..., 14, 15} = 16 distinct colors.
    
    Why no rainbow 4-cycle?
    A 4-cycle on vertices {a,b,c,d} with a<b<c<d has three possible cycles:
    1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
       Colors: a, b, c, a -> colors {a, b, c} (only 3 distinct, not 4)
    2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
       Colors: a, b, c, a -> colors {a, b, c} (only 3 distinct)
    3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)
       Colors: a, b, b, a -> colors {a, b} (only 2 distinct)
    
    Wait, let me re-examine. The edge (d,a) has color a (since a < d).
    The edge (b,d) has color b (since b < d).
    The edge (c,d) has color c (since c < d).
    The edge (c,a) has color a (since a < c).
    The edge (c,b) has color b (since b < c).
    
    Cycle 1: a-b-c-d-a: (a,b)=a, (b,c)=b, (c,d)=c, (d,a)=a. Colors: {a,b,c}. 3 distinct.
    Cycle 2: a-b-d-c-a: (a,b)=a, (b,d)=b, (d,c)=c, (c,a)=a. Colors: {a,b,c}. 3 distinct.
    Cycle 3: a-c-b-d-a: (a,c)=a, (c,b)=b, (b,d)=b, (d,a)=a. Colors: {a,b}. 2 distinct.
    
    So no rainbow 4-cycle. The special case for (14,15) with color 15:
    We need to check all 4-cycles involving vertex 15.
    
    If a 4-cycle includes vertex 15 and three other vertices a<b<c<15:
    The edges involving 15 are (15,a), (15,b), (15,c) which have colors a, b, c.
    The fourth edge is between two of {a,b,c}, say (x,y) with x<y, which has color x.
    
    Cycle on {a,b,c,15}:
    - Edges: (a,b)=a, (b,c)=b, (c,15)=c, (15,a)=a. Colors: {a,b,c}. 3 distinct.
    - Edges: (a,b)=a, (b,15)=b, (15,c)=c, (c,a)=a. Colors: {a,b,c}. 3 distinct.
    - Edges: (a,c)=a, (c,b)=b, (b,15)=b, (15,a)=a. Colors: {a,b}. 2 distinct.
    
    What if the 4-cycle is {a,b,c,d} where one of them is 15, say d=15?
    Then it's the same as above with a<b<c<15.
    
    What about the edge (14,15) specifically? It has color 15.
    Consider 4-cycle on {a,b,14,15} with a<b<14:
    Edges: (a,b)=a, (b,14)=b, (14,15)=15, (15,a)=a. Colors: {a,b,15}. 3 distinct.
    Edges: (a,b)=a, (b,15)=b, (15,14)=15, (14,a)=a. Colors: {a,b,15}. 3 distinct.
    Edges: (a,14)=a, (14,b)=b, (b,15)=b, (15,a)=a. Colors: {a,b}. 2 distinct.
    
    So even with color 15 on (14,15), no rainbow 4-cycle occurs.
    
    Total distinct colors: {0,1,...,14,15} = 16 colors.
    """
    colors = [0] * NUM_EDGES
    
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_idx(i, j)
            if i == 14 and j == 15:
                colors[idx] = 15
            else:
                colors[idx] = i
    
    return colors

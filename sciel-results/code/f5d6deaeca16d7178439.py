"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use two disjoint copies of K_{3,3}.
    - K_{3,3} is bipartite, so girth >= 4 (triangle-free).
    - In K_{3,3}, the minimum degree is 3.
    - The cycle lengths in K_{3,3} are 4 and 6.
    - Thus, the set of distinct cycle lengths is {4, 6}, which has size 2.
    - This is the theoretical minimum for a graph with min degree >= 3 on 12 vertices,
      as a graph with only one cycle length L would require specific structures
      (like a cactus of L-cycles) which cannot satisfy min degree 3 without
      introducing other cycle lengths or violating vertex counts for L=4 or L=6 
      in a way that fits 12 vertices perfectly with min degree 3.
      Specifically, min degree 3 implies the graph is not a simple cycle (degree 2).
      For L=4, a graph with all cycles length 4 and min degree 3 does not exist 
      for n=12 (smallest is likely larger or requires degree > 3).
      For L=5, Petersen (n=10) has 6-cycles too.
      So 2 is the best achievable.
    """
    n = 12
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 in part A, 3,4,5 in part B
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
    
    # Second K_{3,3}: vertices 6,7,8 in part A, 9,10,11 in part B
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
            
    # Ensure types are correct (list of lists of ints)
    # The loop above produces lists of ints.
    
    return edges
# EVOLVE-BLOCK-END

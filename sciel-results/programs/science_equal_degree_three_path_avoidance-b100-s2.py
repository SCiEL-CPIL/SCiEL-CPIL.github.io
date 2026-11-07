import time
import sys

def construct():
    n = 15
    # Strategy: Construct a complete bipartite graph K_7,8.
    # Vertices 0..6 form set A (size 7)
    # Vertices 7..14 form set B (size 8)
    # Degrees in A are 8, Degrees in B are 7.
    # Any simple path of length 3 s-u-v-t in a bipartite graph
    # has s and t in the same partition (if s in A, u in B, v in A, t in B? No.)
    # Let's re-verify path endpoints in bipartite graph.
    # s - u - v - t
    # If s in A, u must be in B.
    # v must be in A (since u in B and v adjacent to u).
    # t must be in B (since v in A and t adjacent to v).
    # So s in A, t in B.
    # deg(s) = 8 (size of B).
    # deg(t) = 7 (size of A).
    # 8 != 7. Condition satisfied.
    #
    # If s in B, u must be in A.
    # v must be in B.
    # t must be in A.
    # So s in B, t in A.
    # deg(s) = 7.
    # deg(t) = 8.
    # 7 != 8. Condition satisfied.
    #
    # Thus K_7,8 is valid.
    # Number of edges = 7 * 8 = 56.
    
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in edges]
    
    return final_edges

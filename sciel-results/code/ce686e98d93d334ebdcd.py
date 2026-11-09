"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges for a graph on 12 vertices (0-11) consisting of two disjoint K_{3,3} components.
    This graph is 3-regular, triangle-free (bipartite), and has girth 4.
    The distinct cycle lengths are 4 and 6, so the raw value is 2.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 (Part A) and 3, 4, 5 (Part B)
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 (Part A') and 9, 10, 11 (Part B')
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END

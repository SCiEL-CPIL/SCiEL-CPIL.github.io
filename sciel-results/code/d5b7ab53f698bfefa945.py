"""Rewritten program: constructs two disjoint K_{3,3} components for optimal raw value of 2."""

def construct():
    # Two disjoint K_{3,3} components on vertices 0-5 and 6-11.
    # K_{3,3} is bipartite (triangle-free), cubic (degree 3), and has
    # simple cycles of exactly lengths 4 and 6.
    # Disconnected union: distinct cycle lengths = {4, 6}, raw value = 2.
    
    edges = [
        # Component 1: K_{3,3} with parts {0,1,2} and {3,4,5}
        [0, 3], [0, 4], [0, 5],
        [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        # Component 2: K_{3,3} with parts {6,7,8} and {9,10,11}
        [6, 9], [6, 10], [6, 11],
        [7, 9], [7, 10], [7, 11],
        [8, 9], [8, 10], [8, 11],
    ]
    
    return edges

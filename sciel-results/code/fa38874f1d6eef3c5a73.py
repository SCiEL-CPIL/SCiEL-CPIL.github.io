"""Rewritten program following the research directive."""
# EVOLVE-BLOCK-START
def construct():
    # Two disjoint K_{3,3} components on vertices 0..5 and 6..11.
    # K_{3,3} is triangle-free, has girth 4, and its simple cycles have
    # lengths exactly 4 and 6 (since max cycle length in K_{n,n} is 2n = 6).
    # Every vertex has degree 3. Disconnected graphs are allowed.
    # Distinct simple cycle lengths: {4, 6} -> count = 2.
    edges = []
    for i in range(3):
        for j in range(3):
            edges.append([i, 3 + j])
            edges.append([6 + i, 9 + j])
    return edges
# EVOLVE-BLOCK-END

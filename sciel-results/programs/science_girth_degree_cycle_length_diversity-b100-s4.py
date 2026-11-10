"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    # Two disjoint K_{3,3} components
    # Component 1: vertices 0,1,2 (part A) and 3,4,5 (part B)
    for a in range(3):
        for b in range(3, 6):
            edges.append([a, b])
    # Component 2: vertices 6,7,8 (part A) and 9,10,11 (part B)
    for a in range(6, 9):
        for b in range(9, 12):
            edges.append([a, b])
    return edges

# EVOLVE-BLOCK-END

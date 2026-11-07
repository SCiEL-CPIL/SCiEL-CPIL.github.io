"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # The optimal graph is K_8,8.
    # Partition vertices into two sets of 8.
    left_partition = [int(i) for i in range(0, 8)]
    right_partition = [int(i) for i in range(8, 16)]
    
    edges = []
    for u in left_partition:
        for v in right_partition:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END

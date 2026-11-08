"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest.
    
    # Helper to add a rank-1 term to U, V, W
    # u: dict (i, j) -> coeff
    # v: dict (j, k) -> coeff
    # w: dict (k, i) -> coeff
    # col: current column index
    def add_term(u, v, w, col):
        for (i, j), val in u.items():
            U[i * m + j][col] += val
        for (j, k), val in v.items():
            V[j * p + k][col] += val
        for (k, i), val in w.items():
            W[k * n + i][col] += val

    # Initialize U, V, W with 0s
    # We don't know the final rank yet, but we can estimate or use a list of columns
    # Let's build the columns as lists of (u, v, w) tuples first.
    columns = []
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen 2x2x2
    # P1
    columns.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # P2
    columns.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1}
    ))
    # P3
    columns.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1}
    ))
    # P4
    columns.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(1,0): 1, (0,0): 1}
    ))
    # P5
    columns.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,1): 1, (0,0): -1}
    ))
    # P6
    columns.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # P7
    columns.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Same Strassen structure, but k indices are 2,3
    # P1
    columns.append((
        {(0,0): 1, (1,1): 1},
        {(0,2): 1, (1,3): 1},
        {(2,0): 1, (3,1): 1}
    ))
    # P2
    columns.append((
        {(1,0): 1, (1,1): 1},
        {(0,2): 1},
        {(3,0): 1}
    ))
    # P3
    columns.append((
        {(0,0): 1},
        {(0,3): 1, (1,3): -1},
        {(2,1): 1}
    ))
    # P4
    columns.append((
        {(1,1): 1},
        {(1,2): 1, (0,2): -1},
        {(3,1): 1, (2,0): 1}
    ))
    # P5
    columns.append((
        {(0,0): 1, (0,1): 1},
        {(1,3): 1},
        {(2,1): 1, (3,0): -1}
    ))
    # P6
    columns.append((
        {(1,0): 1, (0,0): -1},
        {(0,2): 1, (0,3): 1},
        {(3,1): 1}
    ))
    # P7
    columns.append((
        {(0,1): 1, (1,1): -1},
        {(1,2): 1, (1,3): 1},
        {(2,0): 1}
    ))
    
    # Block 3: i=2, j in {0,1}, k in {0,1}
    # Schoolbook for 1x2x2
    for j in [0, 1]:
        for k in [0, 1]:
            columns.append((
                {(2,j): 1},
                {(j,k): 1},
                {(k,2): 1}
            ))
            
    # Block 4: i=2, j in {0,1}, k in {2,3}
    for j in [0, 1]:
        for k in [2, 3]:
            columns.append((
                {(2,j): 1},
                {(j,k): 1},
                {(k,2): 1}
            ))
            
    # Block 5: i in {0,1}, j=2, k in {0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            columns.append((
                {(i,2): 1},
                {(2,k): 1},
                {(k,i): 1}
            ))
            
    # Block 6: i in {0,1}, j=2, k in {2,3}
    for i in [0, 1]:
        for k in [2, 3]:
            columns.append((
                {(i,2): 1},
                {(2,k): 1},
                {(k,i): 1}
            ))
            
    # Block 7: i=2, j=2, k in {0,1}
    for k in [0, 1]:
        columns.append((
            {(2,2): 1},
            {(2,k): 1},
            {(k,2): 1}
        ))
        
    # Block 8: i=2, j=2, k in {2,3}
    for k in [2, 3]:
        columns.append((
            {(2,2): 1},
            {(2,k): 1},
            {(k,2): 1}
        ))

    rank = len(columns)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(columns):
        add_term(u, v, w, c)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using the algebraic construction
    from the extended binary Golay code.
    
    The S(5,6,12) design is the unique Steiner system with these parameters.
    We construct it using the standard method:
    - Start with the 12 points labeled 0-11
    - Use the fact that S(5,6,12) can be constructed from the Golay code
    - The blocks are the supports of weight-6 codewords in the extended Golay code
    
    A more direct construction:
    The Witt design S(5,6,12) has 132 blocks. We can construct it using
    the following known method based on the Mathieu group M12 action.
    
    Key fact: The 132 blocks of S(5,6,12) can be generated from a small
    set of base blocks under the action of a group. However, the most
    reliable method is to use the known explicit construction.
    
    Alternative approach: Use the fact that S(5,6,12) is the unique 5-(12,6,1)
    design. We can construct it by:
    1. Taking the 12 points
    2. Using the extended Golay code (length 24, but we need length 12)
    
    Actually, the cleanest construction uses the following:
    - The 12 points correspond to the 12 vertices of a regular icosahedron or
      equivalently to the projective line over GF(11) plus infinity
    - But a simpler computational approach: use the known fact that the blocks
      of S(5,6,12) can be enumerated from the structure of M12.
    
    Most practical approach for code: Generate all 924 6-subsets, then use
    the exact cover property. But since we know the answer is 132 and it's
    unique, we can use a deterministic construction.
    
    Here's a known construction:
    - Label points as 0,1,2,...,11
    - The blocks can be generated using the following seed blocks and
      the action of a suitable group.
    
    Let me use a different well-known construction:
    The S(5,6,12) design can be constructed from the S(4,5,11) design
    (which is the projective geometry PG(3,3) duality or related to the
    Mathieu group M11).
    
    Actually, the most straightforward computational approach that will
    definitely work: use the known explicit list of blocks, or generate
    them via the Golay code construction.
    
    Let me use this approach: The extended binary Golay code G24 has length 24.
    The S(5,6,12) design is related to the S(5,6,12) subdesign of the
    Golay code. 
    
    A cleaner approach: I'll use the fact that we can construct S(5,6,12)
    by taking the 12 points and using the following:
    
    The 132 blocks of S(5,6,12) are exactly the supports of the weight-6
    codewords in the [12,6,4] code that is the restriction/projection of
    the Golay code. But this is getting complicated.
    
    Let me just use a direct constructive approach:
    
    Method: Use the known construction via the icosahedral group or
    simply enumerate using a clever backtracking with strong pruning.
    
    Actually, the simplest reliable method: since the design is unique and
    small, I can use a deterministic greedy with the right ordering, or
    better yet, use the algebraic construction.
    
    Here's a concrete construction I'll implement:
    
    The S(5,6,12) design can be constructed as follows:
    - Points: Z_11 ∪ {∞} (but we'll label 0-11)
    - Use the fact that M12 acts transitively on the blocks
    
    Let me use a different, very practical approach:
    I'll generate the design using the known property that it's the unique
    5-(12,6,1) design, and construct it by solving the exact cover problem
    with a very efficient algorithm, OR I can hardcode a known construction.
    
    Best approach: Use the construction from the extended Golay code.
    The extended binary Golay code G24 ⊂ F_2^24 has minimum distance 8.
    The S(5,6,12) design is obtained by considering the 12 coordinate
    positions in a specific way... this is getting too complex.
    
    Let me use the most practical approach: a highly optimized backtracking
    search with strong symmetry breaking, which will find the unique solution
    very quickly.
    
    Actually, I recall now: there's a very clean construction.
    
    The 12 points can be identified with the 12 vertices of an icosahedron.
    The 132 blocks correspond to the 66 pairs of opposite faces (each face
    and its antipodal face give 6 vertices? No, that's not right either).
    
    Let me just implement a very fast exact cover solver (Algorithm X with
    Dancing Links) which will find the unique 132-block solution essentially
    instantly for this small problem.
    
    For the exact cover formulation:
    - We have 792 5-subsets to cover
    - We have 924 candidate 6-subsets (blocks)
    - Each block covers exactly 6 of the 5-subsets
    - We need to select blocks such that every 5-subset is covered exactly once
    - This is an exact cover problem: find 132 blocks that cover all 792
      5-subsets exactly once
    
    792 / 6 = 132, so we need exactly 132 blocks.
    
    Algorithm X with dancing links will solve this extremely fast.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    
    # Generate all 5-subsets and all 6-subsets
    all_5subsets = list(combinations(points, 5))
    all_6subsets = list(combinations(points, 6))
    
    # Map each 5-subset to its index
    subset_5_index = {s: i for i, s in enumerate(all_5subsets)}
    num_5subsets = len(all_5subsets)  # 792
    
    # For each 6-subset, compute which 5-subsets it covers
    block_to_5subsets = {}
    for b in all_6subsets:
        subsets_5 = list(combinations(b, 5))
        block_to_5subsets[b] = [subset_5_index[s] for s in subsets_5]
    
    # Build the exact cover matrix
    # Columns: 792 5-subsets
    # Rows: 924 blocks
    # Entry (row, col) = 1 if block 'row' covers 5-subset 'col'
    
    # For dancing links, we need:
    # - For each column, list of rows that have a 1 in that column
    # - For each row, list of columns that have a 1 in that row
    # - For each 1-entry, links to neighboring entries
    
    # Let's use a simpler but still fast approach: Algorithm X with
    # bitsets or with the standard dancing links.
    
    # Actually, for this small problem, let me use a simpler recursive
    # backtracking with strong heuristics.
    
    # Better: use the standard Algorithm X implementation.
    
    # Column lists: for each 5-subset index, which blocks contain it
    col_to_rows = [[] for _ in range(num_5subsets)]
    row_to_cols = {}
    
    for i, b in enumerate(all_6subsets):
        cols = block_to_5subsets[b]
        row_to_cols[i] = cols
        for c in cols:
            col_to_rows[c].append(i)
    
    # Dancing Links implementation
    # We'll use the standard DLX approach.
    
    # Each node in the DLX structure represents a 1 in the matrix.
    # We need:
    # - L, R: left/right pointers (horizontal)
    # - U, D: up/down pointers (vertical)
    # - C: column pointer
    # - Row and Col indices
    
    # For efficiency, let's use arrays.
    
    # First, count the number of 1s
    total_ones = sum(len(row_to_cols[i]) for i in range(len(all_6subsets)))
    
    # Allocate arrays
    L = [0] * (total_ones + num_5subsets + 1)
    R = [0] * (total_ones + num_5subsets + 1)
    U = [0] * (total_ones + num_5subsets + 1)
    D = [0] * (total_ones + num_5subsets + 1)
    C = [0] * (total_ones + num_5subsets + 1)
    size = [0] * (num_5subsets + 1)
    row_of_node = [0] * (total_ones + num_5subsets + 1)
    col_of_node = [0] * (total_5subsets + 1) if False else [0] * (total_ones + num_5subsets + 1)
    
    # Column header nodes: 1 to num_5subsets
    # Root node: 0
    # Data nodes: num_5subsets + 1 to num_5subsets + total_ones
    
    # Initialize column headers
    for i in range(num_5subsets + 1):
        L[i] = i - 1
        R[i] = (i + 1) % (num_5subsets + 1)
        U[i] = i
        D[i] = i
        C[i] = i
        size[i] = len(col_to_rows[i-1]) if i > 0 else 0
    
    # Root is node 0, columns are 1..num_5subsets
    L[0] = num_5subsets
    R[0] = 1
    L[1] = 0
    R[num_5subsets] = 0
    
    # Now add data nodes
    node_id = num_5subsets + 1
    row_nodes = [[] for _ in range(len(all_6subsets))]  # nodes for each row
    
    for r in range(len(all_6subsets)):
        for c in row_to_cols[r]:
            # c is 0-indexed, column header is c+1
            col_header = c + 1
            node = node_id
            node_id += 1
            
            # Set column
            C[node] = col_header
            row_of_node[node] = r
            col_of_node[node] = c
            
            # Insert into column (at the bottom)
            U[node] = U[col_header]
            D[node] = col_header
            D[U[col_header]] = node
            U[col_header] = node
            size[col_header] += 1
            
            # Insert into row
            if row_nodes[r]:
                prev = row_nodes[r][-1]
                L[node] = prev
                R[node] = R[prev]
                L[R[prev]] = node
                R[prev] = node
            else:
                L[node] = node
                R[node] = node
            
            row_nodes[r].append(node)
    
    # Verify node_id
    assert node_id == total_ones + num_5subsets + 1, f"Node count mismatch: {node_id} vs {total_ones + num_5subsets + 1}"
    
    # Solution
    solution = []
    
    def cover(col):
        """Remove column from the matrix."""
        R[L[col]] = R[col]
        L[R[col]] = L[col]
        i = D[col]
        while i != col:
            j = R[i]
            while j != i:
                D[U[j]] = D[j]
                U[D[j]] = U[j]
                size[C[j]] -= 1
                j = R[j]
            i = D[i]
    
    def uncover(col):
        """Restore column to the matrix."""
        i = U[col]
        while i != col:
            j = L[i]
            while j != i:
                size[C[j]] += 1
                D[U[j]] = j
                U[D[j]] = j
                j = L[j]
            i = U[i]
        R[L[col]] = col
        L[R[col]] = col
    
    def search(k):
        """Algorithm X recursive search."""
        if time.time() > deadline:
            return False
        
        if R[0] == 0:
            # No columns left, solution found
            return True
        
        # Choose column with minimum size (S heuristic)
        col = R[0]
        min_size = size[col]
        j = R[col]
        while j != 0:
            if size[j] < min_size:
                min_size = size[j]
                col = j
            j = R[j]
        
        if min_size == 0:
            return False
        
        cover(col)
        r = D[col]
        while r != col:
            solution.append(row_of_node[r])
            
            j = R[r]
            while j != r:
                cover(C[j])
                j = R[j]
            
            if search(k + 1):
                return True
            
            solution.pop()
            
            j = L[r]
            while j != r:
                uncover(C[j])
                j = L[j]
            
            r = D[r]
        
        uncover(col)
        return False
    
    if search(0):
        blocks = [list(all_6subsets[i]) for i in solution]
        # Verify
        assert len(blocks) == 132, f"Expected 132 blocks, got {len(blocks)}"
        return blocks
    else:
        # Fallback: should not happen
        return []

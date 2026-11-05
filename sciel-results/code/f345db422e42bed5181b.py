import itertools
import time

def construct():
    points = range(12)
    all_6sets = list(itertools.combinations(points, 6))
    n_blocks = len(all_6sets)  # 924
    
    all_5sets = list(itertools.combinations(points, 5))
    n_5sets = len(all_5sets)  # 792
    
    five_set_index = {s: i for i, s in enumerate(all_5sets)}
    
    block_5sets = []
    for block in all_6sets:
        subsets = []
        for j in range(6):
            subset = tuple(block[:j] + block[j+1:])
            subsets.append(five_set_index[subset])
        block_5sets.append(subsets)
    
    blocks_by_5set = [[] for _ in range(n_5sets)]
    for b_idx in range(n_blocks):
        for s_idx in block_5sets[b_idx]:
            blocks_by_5set[s_idx].append(b_idx)
    
    # DLX implementation
    # Columns: 0..n_5sets-1
    # Each row (block) has 6 entries
    
    # Build the matrix structure
    # For each column, maintain a list of (row_idx, node_id) in order
    # For each row, maintain list of (col_idx, node_id)
    
    # Node ID: col_idx * n_blocks + row_idx for simplicity? No, let's use a different scheme.
    # Actually, let's use: for column c, the nodes are indexed separately.
    
    # Simpler approach: use arrays
    # col_nodes[c] = list of node IDs in column c (vertical list)
    # row_nodes[r] = list of node IDs in row r (horizontal list)
    # For each node, we need: up, down, left, right, active
    
    # Let's use a flat array approach
    # Total nodes = n_5sets * 6 (each 5-set is in exactly 6 blocks? No, each 5-set is in C(12-5,1)=7 blocks)
    # Actually each 5-set is in 7 blocks (the 6-set plus one more element from the remaining 7)
    # Wait, no. A 5-set {a,b,c,d,e} is contained in 6-sets {a,b,c,d,e,x} for x in the other 7 points.
    # So each 5-set appears in 7 blocks.
    # Total nodes = 792 * 7 = 5544
    
    # Let's rebuild
    # For each 5-set, there are 7 blocks containing it
    # For each block, there are 6 5-sets
    
    # Node structure:
    # We'll have n_5sets columns
    # Each column c has 7 nodes (one for each block containing that 5-set)
    # Each row (block) has 6 nodes (one for each 5-set in that block)
    
    # Let's assign node IDs:
    # For column c, nodes are: c*7 + 0, c*7 + 1, ..., c*7 + 6
    # But we need to know which row each node belongs to
    
    # Actually, let's just build it properly
    # nodes: list of dicts or parallel arrays
    
    # Parallel arrays:
    # up[i], down[i], left[i], right[i], row[i], col[i], active[i]
    
    # First, count nodes per column
    col_counts = [0] * n_5sets
    for b_idx in range(n_blocks):
        for s_idx in block_5sets[b_idx]:
            col_counts[s_idx] += 1
    
    # col_counts should all be 7
    # Total nodes = sum(col_counts) = 792 * 7 = 5544
    
    total_nodes = sum(col_counts)
    
    up = [0] * total_nodes
    down = [0] * total_nodes
    left = [0] * total_nodes
    right = [0] * total_nodes
    row = [0] * total_nodes
    col = [0] * total_nodes
    active = [True] * total_nodes
    
    # Column headers: 0..n_5sets-1 (we'll use separate indices for headers)
    # Actually, let's make headers part of the node array
    # Headers: 0..n_5sets-1
    # Data nodes: n_5sets..n_5sets+total_nodes-1
    
    n_headers = n_5sets
    n_data = total_nodes
    n_total = n_headers + n_data
    
    up = [0] * n_total
    down = [0] * n_total
    left = [0] * n_total
    right = [0] * n_total
    row = [-1] * n_total  # -1 for headers
    col = [0] * n_total
    active = [True] * n_total
    
    # Initialize headers as circular list
    for c in range(n_headers):
        left[c] = (c - 1) % n_headers
        right[c] = (c + 1) % n_headers
        up[c] = c
        down[c] = c
        col[c] = c
    
    # For each column, assign data node IDs
    col_node_start = [0] * n_headers
    idx = n_headers
    for c in range(n_headers):
        col_node_start[c] = idx
        idx += col_counts[c]
    
    # Now fill in the data nodes
    # For each column c, we have col_counts[c] nodes
    # We need to assign them to specific blocks
    
    # For each 5-set c, the blocks containing it are blocks_by_5set[c]
    # There are 7 such blocks
    
    # For each block b, we need to know which node in each of its 6 columns corresponds to it
    
    # Let's create a mapping: for block b, for each of its 6 5-sets, what's the node ID?
    block_node_map = [{} for _ in range(n_blocks)]
    
    for c in range(n_headers):
        blocks = blocks_by_5set[c]
        # blocks has 7 entries
        for i, b in enumerate(blocks):
            node_id = col_node_start[c] + i
            row[node_id] = b
            col[node_id] = c
            block_node_map[b][c] = node_id
    
    # Initialize vertical lists for each column
    for c in range(n_headers):
        count = col_counts[c]
        if count == 0:
            continue
        start = col_node_start[c]
        for i in range(count):
            node_id = start + i
            up[node_id] = start + (i - 1) % count
            down[node_id] = start + (i + 1) % count
        # Header points to first and last
        up[start] = c  # up of first node is header? No, header's down should point to first
        down[c] = start
        up[start] = c
        down[start + count - 1] = c
        up[c] = start + count - 1
    
    # Initialize horizontal lists for each row
    for b in range(n_blocks):
        subsets = block_5sets[b]
        nodes = [block_node_map[b][c] for c in subsets]
        for i in range(len(nodes)):
            left[nodes[i]] = nodes[(i - 1) % len(nodes)]
            right[nodes[i]] = nodes[(i + 1) % len(nodes)]
    
    # DLX cover/uncover
    def cover(c):
        right[left[c]] = right[c]
        left[right[c]] = left[c]
        active[c] = False
        r = down[c]
        while r != c:
            j = right[r]
            while j != r:
                up[down[j]] = up[j]
                down[up[j]] = down[j]
                active[j] = False
                j = right[j]
            r = down[r]
    
    def uncover(c):
        active[c] = True
        r = up[c]
        while r != c:
            j = left[r]
            while j != r:
                active[j] = True
                down[up[j]] = j
                up[down[j]] = j
                j = left[j]
            r = up[r]
        left[right[c]] = c
        right[left[c]] = c
    
    start_time = time.time()
    deadline = start_time + 110
    solution = []
    
    def search():
        if time.time() > deadline:
            return False
        
        # Find column with minimum active rows
        min_size = float('inf')
        min_col = -1
        c = 0
        while c < n_headers:
            if active[c]:
                count = 0
                r = down[c]
                while r != c:
                    if active[r]:
                        count += 1
                    r = down[r]
                if count < min_size:
                    min_size = count
                    min_col = c
                    if count == 0:
                        break
            c += 1
        
        if min_col == -1:
            return len(solution) == 132
        
        if min_size == 0:
            return False
        
        # Try each active row in min_col
        r = down[min_col]
        while r != min_col:
            if active[r]:
                b = row[r]
                solution.append(b)
                
                # Cover all columns in this row
                j = r
                covered = []
                while True:
                    j = right[j]
                    if j == r:
                        break
                    if active[j]:
                        cover(col[j])
                        covered.append(col[j])
                
                if search():
                    return True
                
                for c in reversed(covered):
                    uncover(c)
                
                solution.pop()
            
            r = down[r]
        
        return False
    
    # Symmetry breaking: try each first block containing {0,1,2,3,4}
    first_block_candidates = []
    for x in range(5, 12):
        target = tuple(sorted([0, 1, 2, 3, 4, x]))
        for b in range(n_blocks):
            if all_6sets[b] == target:
                first_block_candidates.append(b)
                break
    
    for fb in first_block_candidates:
        if time.time() > deadline:
            break
        
        solution = [fb]
        fb_5sets = block_5sets[fb]
        for c in fb_5sets:
            cover(c)
        
        found = search()
        
        for c in fb_5sets:
            uncover(c)
        
        if found:
            return [list(all_6sets[i]) for i in solution]
    
    return []

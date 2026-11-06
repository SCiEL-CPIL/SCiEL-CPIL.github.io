def construct():
    import itertools
    
    points = list(range(12))
    all_5_subsets = list(itertools.combinations(points, 5))
    all_6_subsets = list(itertools.combinations(points, 6))
    
    n_cols = len(all_5_subsets)
    n_rows = len(all_6_subsets)
    
    subset_index = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Build adjacency
    row_cols = [[] for _ in range(n_rows)]
    col_rows = [[] for _ in range(n_cols)]
    
    for r_idx, block in enumerate(all_6_subsets):
        for i in range(6):
            sub = tuple(x for j, x in enumerate(block) if j != i)
            c_idx = subset_index[sub]
            row_cols[r_idx].append(c_idx)
            col_rows[c_idx].append(r_idx)
            
    # DLX Implementation
    # We use a simple recursive DLX with arrays.
    # Total nodes: n_rows * 6 (each row has 6 ones) + n_cols (headers) + 1 (super header)
    # Max nodes ~ 5544 + 792 + 1 ~ 6337
    
    MAX_NODES = 10000
    L = [0] * MAX_NODES
    R = [0] * MAX_NODES
    U = [0] * MAX_NODES
    D = [0] * MAX_NODES
    C = [0] * MAX_NODES
    Row = [0] * MAX_NODES
    Size = [0] * (n_cols + 1)
    
    node_count = 0
    
    # Initialize header row
    # Node 0 is the super header
    for i in range(n_cols + 1):
        L[i] = i - 1
        R[i] = (i + 1) % (n_cols + 1)
        U[i] = i
        D[i] = i
        C[i] = i
        Size[i] = 0
        if i > 0:
            Row[i] = -1 # Header
        node_count += 1
        
    # Add rows
    for r_idx in range(n_rows):
        cols = row_cols[r_idx]
        if not cols: continue
        first = node_count
        for c_idx in cols:
            # Create node
            node = node_count
            node_count += 1
            
            C[node] = c_idx
            Row[node] = r_idx
            
            # Add to column c_idx
            Size[c_idx] += 1
            U[node] = U[c_idx]
            D[node] = c_idx
            D[U[c_idx]] = node
            U[c_idx] = node
            
            # Add to row r_idx
            if node == first:
                L[node] = node
                R[node] = node
            else:
                prev = R[first]
                L[node] = prev
                R[node] = first
                R[prev] = node
                L[first] = node
                
    # DLX Functions
    def remove(node):
        L[R[node]] = L[node]
        R[L[node]] = R[node]
        i = D[node]
        while i != node:
            L[R[i]] = L[i]
            R[L[i]] = R[i]
            Size[C[i]] -= 1
            i = D[i]
            
    def restore(node):
        i = U[node]
        while i != node:
            Size[C[i]] += 1
            L[R[i]] = i
            R[L[i]] = i
            i = U[i]
        L[R[node]] = node
        R[L[node]] = node
        
    solution = []
    
    def search(k):
        if k > 132:
            return False
        if R[0] == 0:
            # No columns left
            if k == 132:
                return True
            else:
                return False
                
        # Select column with minimum size (S heuristic)
        c = R[0]
        min_size = Size[c]
        while c != 0:
            if Size[c] < min_size:
                min_size = Size[c]
                c = R[c]
        # If min_size is 0, no solution
        if min_size == 0:
            return False
            
        remove(c)
        r = D[c]
        while r != c:
            solution.append(Row[r])
            j = R[r]
            while j != r:
                remove(D[j])
                j = R[j]
                
            if search(k + 1):
                return True
                
            solution.pop()
            j = L[r]
            while j != r:
                restore(U[j])
                j = L[j]
            restore(c)
            r = D[r]
            
        return False
        
    if search(0):
        return [list(all_6_subsets[r]) for r in solution]
    else:
        return []


import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    # 1. Generate all possible blocks (6-subsets) and 5-subsets
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)  # 924
    
    all_5_subsets = list(combinations(range(n_points), subset_size))
    num_5_subsets = len(all_5_subsets)  # 792
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Build the exact cover matrix for DLX
    # Each block covers 6 five-subsets
    # We represent the matrix as a list of lists of column indices for each row
    # row[i] is a list of column indices (5-subset indices) covered by block i
    
    # DLX implementation
    # We use a dictionary-based or array-based implementation.
    # Array-based is faster but requires careful indexing.
    # Let's use a standard DLX implementation with arrays.
    
    # Total columns: num_5_subsets + 1 (header)
    # Total rows: num_blocks + 1 (header)
    # Each node has: L, R, U, D, row, col
    
    # We can optimize by using a sparse representation or just the standard DLX.
    # Given the size (924 x 792), standard DLX is fine.
    
    # Let's implement a lightweight DLX solver.
    
    # Number of nodes in the matrix
    # Each block has 6 non-zero entries.
    # Total nodes = num_blocks * 6
    num_nodes = num_blocks * 6
    
    # We need arrays for L, R, U, D, row, col
    # Size: num_nodes + num_5_subsets + 1 (for header row and columns)
    # Actually, let's just allocate enough space.
    # Max nodes = num_blocks * 6 + num_5_subsets + 1
    max_nodes = num_blocks * 6 + num_5_subsets + 1
    
    L = [-1] * max_nodes
    R = [-1] * max_nodes
    U = [-1] * max_nodes
    D = [-1] * max_nodes
    row_id = [-1] * max_nodes
    col_id = [-1] * max_nodes
    
    # Header node is 0
    # Column headers are 1 to num_5_subsets
    # Data nodes start from num_5_subsets + 1
    
    # Initialize header row
    # Header nodes for columns 0..num_5_subsets-1 (shifted by 1 in 1-based indexing for DLX)
    # Let's use 0-based for columns in the logic, but 1-based for DLX nodes to avoid 0.
    # Header node: 0
    # Column j header node: j + 1
    # Data nodes: start from num_5_subsets + 1
    
    # Link header nodes in a circular list
    L[0] = num_5_subsets
    R[0] = 1
    for j in range(num_5_subsets):
        node = j + 1
        L[node] = node - 1
        R[node] = node + 1
        U[node] = node
        D[node] = node
        col_id[node] = j
        row_id[node] = -1 # Header
    
    R[num_5_subsets] = 0
    L[1] = 0
    
    # Column size counters
    col_size = [0] * num_5_subsets
    
    next_node = num_5_subsets + 1
    
    # Fill the matrix
    for i, b in enumerate(all_blocks):
        subsets_in_b = combinations(b, subset_size)
        # We need to add nodes for each subset in this block
        # We will link them vertically in the column lists
        
        first_node_in_row = -1
        for s in subsets_in_b:
            idx = subset_to_idx[s]
            col_header = idx + 1
            
            # Create new node
            node = next_node
            next_node += 1
            
            row_id[node] = i
            col_id[node] = idx
            
            # Link vertically into column
            U[node] = U[col_header]
            D[node] = col_header
            D[U[col_header]] = node
            U[col_header] = node
            
            # Update column size
            col_size[idx] += 1
            
            # Link horizontally in row
            if first_node_in_row == -1:
                first_node_in_row = node
                L[node] = node
                R[node] = node
            else:
                R[node] = R[first_node_in_row]
                L[node] = first_node_in_row
                R[first_node_in_row] = node
                L[R[node]] = node
    
    # Solution array
    solution = []
    
    def remove(node):
        R[L[node]] = R[node]
        L[R[node]] = L[node]
        
    def restore(node):
        L[R[node]] = node
        R[L[node]] = node
        
    def search(k):
        if time.time() > deadline:
            return False
            
        if R[0] == 0:
            # All columns covered
            return True
            
        # Choose column with minimum size (MRV)
        c = R[0]
        min_size = col_size[0]
        j = c
        while j != 0:
            if col_size[col_id[j]] < min_size:
                min_size = col_size[col_id[j]]
                c = j
            j = R[j]
            
        if min_size == 0:
            return False
            
        # Cover column c
        remove(c)
        i = D[c]
        while i != c:
            solution.append(row_id[i])
            
            j = R[i]
            while j != i:
                remove(j)
                k2 = D[j]
                while k2 != j:
                    col_size[col_id[k2]] -= 1
                    remove(k2)
                    k2 = D[k2]
                j = R[j]
                
            if search(k + 1):
                return True
                
            solution.pop()
            
            j = L[i]
            while j != i:
                k2 = U[j]
                while k2 != j:
                    col_size[col_id[k2]] += 1
                    restore(k2)
                    k2 = U[k2]
                restore(j)
                j = L[j]
                
            i = D[i]
            
        restore(c)
        
        return False
        
    # Run search
    if search(0):
        # Extract solution
        # solution contains row indices (block indices)
        result = [list(all_blocks[i]) for i in solution]
        return result
    else:
        return []

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)
    
    all_5_subsets = list(combinations(range(n_points), subset_size))
    num_5_subsets = len(all_5_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # DLX Setup
    max_nodes = num_blocks * 6 + num_5_subsets + 1
    
    L = [-1] * max_nodes
    R = [-1] * max_nodes
    U = [-1] * max_nodes
    D = [-1] * max_nodes
    row_id = [-1] * max_nodes
    col_id = [-1] * max_nodes
    
    L[0] = num_5_subsets
    R[0] = 1
    for j in range(num_5_subsets):
        node = j + 1
        L[node] = node - 1
        R[node] = node + 1
        U[node] = node
        D[node] = node
        col_id[node] = j
        row_id[node] = -1
    
    R[num_5_subsets] = 0
    L[1] = 0
    
    col_size = [0] * num_5_subsets
    
    next_node = num_5_subsets + 1
    
    for i, b in enumerate(all_blocks):
        subsets_in_b = combinations(b, subset_size)
        first_node_in_row = -1
        for s in subsets_in_b:
            idx = subset_to_idx[s]
            col_header = idx + 1
            
            node = next_node
            next_node += 1
            
            row_id[node] = i
            col_id[node] = idx
            
            U[node] = U[col_header]
            D[node] = col_header
            D[U[col_header]] = node
            U[col_header] = node
            
            col_size[idx] += 1
            
            if first_node_in_row == -1:
                first_node_in_row = node
                L[node] = node
                R[node] = node
            else:
                R[node] = R[first_node_in_row]
                L[node] = first_node_in_row
                R[first_node_in_row] = node
                L[R[node]] = node
    
    solution = []
    
    def remove(node):
        R[L[node]] = R[node]
        L[R[node]] = L[node]
        
    def restore(node):
        L[R[node]] = node
        R[L[node]] = node
        
    def search(k):
        if time.time() > deadline:
            return False
            
        if R[0] == 0:
            return True
            
        c = R[0]
        min_size = col_size[0]
        j = c
        while j != 0:
            if col_size[col_id[j]] < min_size:
                min_size = col_size[col_id[j]]
                c = j
            j = R[j]
            
        if min_size == 0:
            return False
            
        remove(c)
        i = D[c]
        while i != c:
            solution.append(row_id[i])
            
            j = R[i]
            while j != i:
                remove(j)
                k2 = D[j]
                while k2 != j:
                    col_size[col_id[k2]] -= 1
                    remove(k2)
                    k2 = D[k2]
                j = R[j]
                
            if search(k + 1):
                return True
                
            solution.pop()
            
            j = L[i]
            while j != i:
                k2 = U[j]
                while k2 != j:
                    col_size[col_id[k2]] += 1
                    restore(k2)
                    k2 = U[k2]
                restore(j)
                j = L[j]
                
            i = D[i]
            
        restore(c)
        return False
        
    if search(0):
        result = [list(all_blocks[i]) for i in solution]
        return result
    else:
        return []
# EVOLVE-BLOCK-END

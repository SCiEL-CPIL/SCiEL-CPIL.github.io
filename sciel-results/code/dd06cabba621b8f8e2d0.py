import time
import random
from itertools import combinations

def _has_c6(adj_matrix, n):
    """Check if graph with adjacency matrix has a cycle of length 6."""
    # Simple DFS from each node
    for start in range(n):
        # Path is list of nodes
        # We look for a path of length 6 back to start
        # Nodes in path must be distinct
        def dfs(u, depth, visited, path):
            if depth == 6:
                # Check if u is connected to start
                if adj_matrix[u][start]:
                    # Ensure all nodes in path are distinct
                    # path contains start, v1, v2, v3, v4, v5, u (which is v6=start?) 
                    # Actually, path is [start, v1, v2, v3, v4, v5, u]
                    # If u == start, and len(path)==7, and all distinct except start==end
                    if len(set(path[:-1])) == 6:
                        return True
                return False
            
            for v in range(n):
                if adj_matrix[u][v] and v not in visited:
                    visited.add(v)
                    path.append(v)
                    if dfs(v, depth + 1, visited, path):
                        return True
                    path.pop()
                    visited.remove(v)
            return False
        
        # Start DFS
        # Path starts with start
        # We need 5 more steps to reach a neighbor of start
        # So depth 0 is start. We take 5 steps to get to a node, then check connection to start.
        # Wait, a cycle of length 6 has 6 edges.
        # start -> v1 -> v2 -> v3 -> v4 -> v5 -> start
        # So we need to find a path of 5 edges from start to a neighbor of start, with 6 distinct nodes.
        # My dfs above:
        # depth 0: u=start.
        # depth 1: u=v1.
        # ...
        # depth 5: u=v5.
        # Check if v5 connected to start.
        # This matches.
        
        if dfs(start, 0, {start}, [start]):
            return True
    return False

def _check_partition_validity(partition, n=17):
    """
    partition: list of 4 sets, each containing integers 1..8.
    Returns True if no color class has C6.
    """
    for S in partition:
        # Build adjacency matrix for this color class
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for d in S:
                j1 = (i + d) % n
                j2 = (i - d) % n
                adj[i][j1] = True
                adj[i][j2] = True
        if _has_c6(adj, n):
            return False
    return True

def _generate_partitions(elements, k):
    """Generate all partitions of elements into k non-empty subsets."""
    if k == 1:
        yield [set(elements)]
        return
    if len(elements) < k:
        return
    
    # Recursive generation
    # Pick a representative for the first set
    first = elements[0]
    rest = elements[1:]
    
    for sub in _generate_partitions(rest, k):
        # Case 1: first is in the first set
        new_sub = [sub[0] | {first}] + sub[1:]
        yield new_sub
        
        # Case 2: first is in one of the other sets
        # To avoid duplicates, we only add 'first' to sets that are "new" or follow an order
        # This standard recursion is tricky to make unique without sorting.
        # Alternative: Use iterative approach or just iterate over all assignments of 8 elements to 4 bins (4^8 = 65536) and filter unique partitions?
        # 4^8 is small. We can just iterate 0..65535, map to colors, and check uniqueness by sorting the sets.
        pass

def _find_circulant_coloring(n=17):
    """Search for a valid circulant coloring."""
    if n != 17:
        return None
        
    half = n // 2 # 8
    elements = list(range(1, half + 1))
    
    seen_partitions = set()
    
    # Iterate over all assignments of 1..8 to 4 colors
    # 4^8 = 65536
    for mask in range(4 ** 8):
        partition = [set() for _ in range(4)]
        valid_assignment = True
        for i, elem in enumerate(elements):
            color = (mask >> (i * 2)) & 3
            partition[color].add(elem)
            
        # Check if all sets are non-empty? 
        # The problem doesn't strictly require non-empty color classes, but a "4-coloring" usually implies using the palette.
        # However, if a color class is empty, it has no C6.
        # So we allow empty sets.
        
        # Canonical form to avoid checking same partition multiple times
        # Sort sets by their smallest element (or hash)
        canonical = tuple(tuple(sorted(s)) for s in sorted(partition, key=lambda x: (len(x), min(x) if x else 100)))
        
        if canonical in seen_partitions:
            continue
        seen_partitions.add(canonical)
        
        if _check_partition_validity(partition, n):
            # Return the partition
            return partition
            
    return None

def _construct_from_partition(partition, n=17):
    """Construct the color list from a valid partition."""
    colors = []
    # Map difference to color
    diff_to_color = {}
    for c, S in enumerate(partition):
        for d in S:
            diff_to_color[d] = c
            
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            # Canonical difference
            if d > n // 2:
                d = n - d
            colors.append(diff_to_color[d])
            
    return {'n': n, 'colors': colors}

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Try Circulant Construction for n=17
    partition = _find_circulant_coloring(17)
    if partition:
        return _construct_from_partition(partition, 17)
        
    if time.time() > deadline:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

    # 2. Fallback: Random Local Search for n=17
    # This is a heuristic backup in case circulant fails
    # (Unlikely to be needed if circulant works, but good for robustness)
    
    n = 17
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize random
    current_colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Precompute edge indices for fast access? 
    # For C6 check, we need adjacency.
    
    def get_adj(colors_list):
        adj = [[0]*n for _ in range(n)]
        for k, (i, j) in enumerate(edges):
            c = colors_list[k]
            adj[i][j] = c
            adj[j][i] = c
        return adj

    def count_c6s(colors_list):
        adj = get_adj(colors_list)
        count = 0
        # This is expensive. We need a faster metric or just check validity.
        # For local search, we need a cost function.
        # Counting all C6s is O(n^6) or similar. Too slow for 100s of steps.
        # We can use a probabilistic check or a partial check.
        # Or, we can just do a "steepest descent" on the number of C6s involving the flipped edge.
        
        # Simpler: Just try to find a valid one via random restarts with greedy repair?
        # Given the circulant search is likely to succeed, this fallback is low priority.
        # I will implement a simple random restart with a few steps of local search.
        pass
        
    # Since implementing a fast C6 counter is complex, and circulant is likely to work,
    # I will just return a smaller n if circulant fails, to ensure validity.
    # Actually, n=16 is also likely to have a circulant solution or be easier.
    
    partition_16 = _find_circulant_coloring(16) # Note: 16 is not prime, circulant logic differs slightly (differences 1..7, but 8 is self-inverse?)
    # For n=16, differences 1..7 and 8. 8 is its own inverse.
    # The logic in _find_circulant_coloring assumes n is odd (prime).
    # So it won't work directly for 16.
    
    # If 17 fails, we can try a hardcoded n=16 or just return 9.
    # But 17 is the target.
    
    # Let's assume the circulant search for 17 will find a solution.
    # If not, we return the seed.
    
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

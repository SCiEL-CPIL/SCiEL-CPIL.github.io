import time
import random
import numpy as np
from itertools import combinations

def _has_c6(adj_matrix, n):
    """Check if graph with adjacency matrix has a cycle of length 6."""
    # Simple DFS from each node
    for start in range(n):
        # Path is list of nodes
        # We look for a path of length 6 back to start
        # Nodes in path must be distinct
        def dfs(u, depth, visited, path):
            if depth == 5:
                # Check if u is connected to start
                if adj_matrix[u][start]:
                    # Ensure all nodes in path are distinct
                    # path contains start, v1, v2, v3, v4, v5
                    # If u is connected to start, and u is not start (handled by visited),
                    # and all 6 nodes are distinct, it's a C6.
                    # visited contains start, v1...v4. u is v5.
                    # So nodes are start, v1, v2, v3, v4, v5. 6 distinct nodes.
                    if len(set(path)) == 6:
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

def _precompute_path4(v, adj_list):
    """
    Precompute Path4[c][i][d] for all c, i, d < v.
    Path4[c][i][d] is True if there is a simple path of length 4 
    between i and d in the graph G_c induced by vertices 0..v-1.
    """
    Path4 = [ [[False]*v for _ in range(v)] for _ in range(4) ]
    for c in range(4):
        for i in range(v):
            if not adj_list[c][i]:
                continue
            for d in range(v):
                if i == d:
                    continue
                found = False
                # Path i-a-b-e-d
                # a is neighbor of i
                for a in adj_list[c][i]:
                    if a == d:
                        continue
                    # b is neighbor of a
                    for b in adj_list[c][a]:
                        if b == i or b == d or b == a:
                            continue
                        # e is neighbor of b
                        for e in adj_list[c][b]:
                            if e == i or e == d or e == a or e == b:
                                continue
                            # Check if e is connected to d
                            if d in adj_list[c][e]:
                                found = True
                                break
                        if found:
                            break
                    Path4[c][i][d] = found
    return Path4

def _constrained_backtrack(n_target, deadline):
    adj = [[-1]*n_target for _ in range(n_target)]
    adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
    
    def _search(v):
        if time.time() > deadline:
            return False
        
        if v == n_target:
            return True
        
        if v == 0:
            return _search(v + 1)
        
        # Precompute Path4 for the current subgraph 0..v-1
        # This is done once per level v
        Path4 = _precompute_path4(v, adj_list)
        
        def _assign(idx):
            if time.time() > deadline:
                return False
            
            if idx == v:
                return _search(v + 1)
            
            i = idx
            for c in range(4):
                # Check if setting adj[i][v]=c creates a C6
                # A C6 would involve v, i, and 4 other vertices in a path i-a-b-e-d
                # where d is another neighbor of v in color c.
                valid = True
                for d in range(i):
                    if adj[d][v] == c:
                        if Path4[c][i][d]:
                            valid = False
                            break
                if not valid:
                    continue
                
                adj[i][v] = c
                adj[v][i] = c
                adj_list[c][i].add(v)
                adj_list[c][v].add(i)
                
                if _assign(idx + 1):
                    return True
                
                adj[i][v] = -1
                adj[v][i] = -1
                adj_list[c][i].remove(v)
                adj_list[c][v].remove(i)
            
            return False
        
        return _assign(0)
    
    if _search(0):
        colors = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                colors.append(adj[i][j])
        return colors
    return None

def _verify(colors, n):
    if n < 6:
        return True
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    
    # Re-verify using the same Path4 logic or a simple check
    # For safety, use a brute force check on the final result
    adj_list = [[set() for _ in range(n)] for _ in range(4)]
    for i in range(n):
        for j in range(i+1, n):
            c = adj[i][j]
            adj_list[c][i].add(j)
            adj_list[c][j].add(i)
    
    for c in range(4):
        # Check for C6 in color c
        # We can reuse _precompute_path4 logic or just check all pairs
        # If there is a C6, there exists a pair (i, d) with a path of length 4
        # But wait, the C6 is a cycle, so i and d are connected by the path AND by the cycle closure?
        # No, the cycle is v-u1-u2-u3-u4-u5-v.
        # In the final graph, we just need to check if any color class contains a C6.
        # My Path4 check was for "path of length 4 between two neighbors of v".
        # This forms a C6 when v is added.
        # In the final graph, a C6 is any 6-cycle.
        # We can check if there exists any i, d such that Path4[c][i][d] is true 
        # AND i, d are NOT connected by an edge? No.
        # If Path4[c][i][d] is true, there is a path i-a-b-e-d.
        # If i and d are connected by an edge, it's a C5 or C4 etc.
        # If i and d are NOT connected, it's just a path.
        # To form a C6, we need a path of length 5 between two adjacent vertices?
        # Or a path of length 4 between two vertices that are connected?
        # No, a C6 is a cycle of length 6.
        # It consists of 6 edges.
        # If we remove one edge (u, v), we have a path of length 5 between u and v.
        # So, a C6 exists iff there exist adjacent vertices u, v in G_c 
        # such that there is a simple path of length 5 between u and v in G_c - (u, v).
        
        # Let's just do a direct check for C6 in the final graph to be safe.
        # n is at most 20. C(20, 6) = 38760.
        # For each subset, check if it contains a C6.
        # This is fast enough for verification.
        
        for subset in combinations(range(n), 6):
            # Check if the induced subgraph on subset contains a C6
            # A 6-vertex graph contains a C6 iff it has a Hamiltonian cycle.
            # There are 60 Hamiltonian cycles on 6 labeled vertices.
            # We can check if any of them is present.
            
            # Precompute the 60 cycles on 0..5
            if not hasattr(_verify, '_cycles'):
                cycles = []
                seen = set()
                from itertools import permutations
                for perm in permutations(range(1, 6)):
                    cycle = (0,) + perm
                    edges = frozenset(frozenset((cycle[k], cycle[(k+1)%6])) for k in range(6))
                    if edges not in seen:
                        seen.add(edges)
                        cycles.append(cycle)
                _verify._cycles = cycles
            
            for cycle in _verify._cycles:
                ok = True
                for k in range(6):
                    u = subset[cycle[k]]
                    w = subset[cycle[(k+1)%6]]
                    if adj[u][w] != c:
                        ok = False
                        break
                if ok:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Try Circulant Construction for n=17
    partition = _find_circulant_coloring(17)
    if partition:
        result = _construct_from_partition(partition, 17)
        if _verify(result['colors'], 17):
            return result
            
    if time.time() > deadline:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

    # 2. Fallback: Constrained Backtrack for n=17
    # This is a heuristic backup in case circulant fails
    # (Unlikely to be needed if circulant works, but good for robustness)
    
    colors = _constrained_backtrack(17, deadline)
    if colors:
        if _verify(colors, 17):
            return {'n': 17, 'colors': colors}
            
    if time.time() > deadline:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

    # 3. Try n=16
    # Circulant for 16 is tricky, so use backtrack
    colors = _constrained_backtrack(16, deadline)
    if colors:
        if _verify(colors, 16):
            return {'n': 16, 'colors': colors}
            
    if time.time() > deadline:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

    # 4. Try n=15
    colors = _constrained_backtrack(15, deadline)
    if colors:
        if _verify(colors, 15):
            return {'n': 15, 'colors': colors}
            
    # Fallback to seed
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

if __name__ == '__main__':
    res = construct()
    print(res)

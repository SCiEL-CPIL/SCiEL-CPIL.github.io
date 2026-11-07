import time
import sys
from collections import deque
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def _build_adjacency(tree_edges, k):
    """Build adjacency list for a tree."""
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _find_root_order(adj, k):
    """Find a BFS/DFS order from a leaf root for embedding."""
    # Find a leaf to use as root
    root = 0
    for v in range(k):
        if len(adj[v]) == 1 and k > 1:
            root = v
            break
    # BFS order
    visited = [False] * k
    visited[root] = True
    order = [root]
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                order.append(v)
                queue.append(v)
    return order, root


def _embed_tree(tree_edges, k, used_edges, n=20, deadline=None, max_backtracks=500000):
    """
    Embed tree of size k into K_n using only unused edges.
    Returns a list of host vertices (length k) or None.
    used_edges: set of (min(a,b), max(a,b)) tuples of used edges.
    """
    if k == 1:
        # Just pick any free vertex
        for v in range(n):
            return [v]
        return None
    
    if k == 2:
        for u in range(n):
            for v in range(u+1, n):
                e = (u, v) if u < v else (v, u)
                if e not in used_edges:
                    return [u, v]
        return None
    
    adj = _build_adjacency(tree_edges, k)
    order, root = _find_root_order(adj, k)
    
    # Parent mapping from order
    parent = [-1] * k
    # Reconstruct parent from BFS
    visited = [False] * k
    visited[root] = True
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
    
    # Map: position in order -> tree vertex
    # order[0] = root, order[i] has parent order[j] for some j < i
    # We need: for each position i in order, what's the parent position?
    pos_of_vertex = [0] * k
    for i, v in enumerate(order):
        pos_of_vertex[v] = i
    
    # For each position i (i>=1), parent position
    parent_pos = [-1] * len(order)
    for i in range(1, len(order)):
        v = order[i]
        p = parent[v]
        parent_pos[i] = pos_of_vertex[p]
    
    # Backtracking search
    host = [-1] * k
    backtracks = 0
    
    def dfs(pos):
        nonlocal backtracks
        if deadline is not None and time.time() > deadline:
            return False
        if pos == k:
            return True
        v = order[pos]
        if pos == 0:
            # Root: try all vertices
            candidates = list(range(n))
        else:
            pp = parent_pos[pos]
            hv = host[pp]
            candidates = [u for u in range(n) if u != hv and (min(hv,u), max(hv,u)) not in used_edges]
        
        # Shuffle candidates for diversity? No, deterministic first.
        for u in candidates:
            if pos == 0:
                host[pos] = u
            else:
                pp = parent_pos[pos]
                hv = host[pp]
                if u == hv:
                    continue
                e = (min(hv, u), max(hv, u))
                if e in used_edges:
                    continue
                host[pos] = u
            
            if dfs(pos + 1):
                return True
            
            host[pos] = -1
            backtracks += 1
            if backtracks > max_backtracks:
                return False
        
        return False
    
    if dfs(0):
        # Convert order-based host to vertex-based host
        result = [0] * k
        for i, v in enumerate(order):
            result[v] = host[i]
        return result
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Sort by size descending, but keep track of original index
    # T1 has 0 edges, T2 has 1 edge, T_k has k-1 edges for k>=1
    # Actually T1: 0 edges, T2: 1 edge, T3: 2 edges, ..., T20: 19 edges
    # Embed largest first (hardest to fit)
    order = list(range(20))  # index 0..19 corresponds to T1..T20
    # Sort by number of edges descending (i.e., by k descending)
    order.sort(key=lambda i: -(i + 1))  # T_{i+1} has i edges
    
    used_edges = set()
    result = [None] * 20
    
    for idx in order:
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            # Single vertex, no edges
            # Pick any vertex
            for v in range(n):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            # Single edge
            for u in range(n):
                for v in range(u + 1, n):
                    e = (u, v)
                    if e not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add(e)
                        break
                if result[idx] is not None:
                    break
            continue
        
        # Larger trees: use backtracking
        mapping = _embed_tree(tree_edges, k, used_edges, n, deadline, max_backtracks=200000)
        
        if mapping is not None:
            # Verify and add edges
            result[idx] = mapping
            for u, v in tree_edges:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                if e in used_edges:
                    # Conflict! Should not happen
                    result[idx] = None
                    break
                used_edges.add(e)
        else:
            result[idx] = None
    
    # Check if we ran out of time
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    # Ensure all entries are proper lists of Python ints or None
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result

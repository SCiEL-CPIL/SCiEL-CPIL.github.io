"""Packing trees T1..T20 into K20 via randomized backtracking with edge-conflict checks."""
import time
from random import Random, shuffle

# Generate the fixed trees exactly as specified
def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def _build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in adj:
        adj[v] = sorted(adj[v])
    return adj

def _embed_tree(tree_edges, n, used_edges_set, deadline, rng):
    """
    Find an embedding of the tree into K_n such that no edge conflicts with used_edges_set.
    Returns a list mapping tree-vertex -> host-vertex, or None if not found.
    """
    if not tree_edges:
        # Single vertex tree (k=1): any free vertex works
        return [0]

    k = len(tree_edges) + 1  # number of vertices in tree
    adj = _build_adjacency(tree_edges)
    
    # Find a root (vertex with highest degree)
    root = max(range(k), key=lambda v: len(adj.get(v, [])))
    
    # Build parent/children structure by BFS from root
    parent = [-1] * k
    children = [[] for _ in range(k)]
    order = []  # BFS order
    visited = {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for nb in adj.get(v, []):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = v
                children[v].append(nb)
                queue.append(nb)
    
    # We'll do randomized backtracking
    # Try multiple random root assignments and random orderings of candidate hosts
    
    # For each tree vertex, precompute which host vertices it could potentially map to
    # (this is just 0..n-1 minus used, but we'll filter during search)
    
    # Strategy: assign root first, then BFS children in random order
    # Use forward checking: after assigning a vertex, check that each unassigned child
    # still has at least one feasible host
    
    def try_embed(root_host):
        map_tree = [-1] * k
        used_hosts = set()
        map_tree[root] = root_host
        used_hosts.add(root_host)
        
        # Order of assignment: BFS order excluding root, but we'll randomize children order
        # Actually, let's use the BFS order but randomize the candidate host order
        order_no_root = [v for v in order if v != root]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order_no_root):
                return True
            v = order_no_root[idx]
            p = parent[v]
            
            # Candidate hosts: all free hosts that don't conflict with parent edge
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                a, b = map_tree[p], h
                if (a, b) in used_edges_set or (b, a) in used_edges_set:
                    continue
                candidates.append(h)
            
            # Randomize candidate order
            shuffle(candidates)
            
            for h in candidates:
                # Forward checking: for each child of v, ensure at least one feasible host exists
                feasible = True
                for ch in children[v]:
                    found = False
                    for h2 in range(n):
                        if h2 == h or h2 in used_hosts:
                            continue
                        if (h, h2) in used_edges_set or (h2, h) in used_edges_set:
                            continue
                        found = True
                        break
                    if not found:
                        feasible = False
                        break
                if not feasible:
                    continue
                
                map_tree[v] = h
                used_hosts.add(h)
                if backtrack(idx + 1):
                    return True
                used_hosts.remove(h)
                map_tree[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Try different root hosts in random order
    root_hosts = list(range(n))
    shuffle(root_hosts)
    
    for root_host in root_hosts:
        if time.time() > deadline:
            return None
        if try_embed(root_host):
            # Extract the mapping
            return [map_tree[v] for v in range(k)]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 20
    trees = _instance_trees(n, 611)
    
    # Process trees in order of decreasing size (most constrained first)
    # Tree index i has (i+1) vertices and i edges (for i>=1), tree 0 has 0 edges
    tree_indices = list(range(n))
    tree_indices.sort(key=lambda i: -i)  # larger i = more edges
    
    used_edges_set = set()
    result = [None] * n
    total_edges = 0
    
    # Use a single RNG for all randomization
    rng = Random(42)
    
    for idx in tree_indices:
        if time.time() > deadline:
            break
        tree_edges = trees[idx]
        if not tree_edges:
            # T1: single vertex, always embeddable
            result[idx] = [0]
            continue
        
        # Try multiple times with different random seeds if needed
        embedding = None
        for attempt in range(50):
            if time.time() > deadline:
                break
            embedding = _embed_tree(tree_edges, n, used_edges_set, deadline, rng)
            if embedding is not None:
                break
        
        if embedding is None:
            # Try with a fresh RNG
            rng2 = Random(attempt + 1000)
            embedding = _embed_tree(tree_edges, n, used_edges_set, deadline, rng2)
        
        if embedding is None:
            continue
        
        # Add edges to used set
        for a, b in tree_edges:
            ha, hb = embedding[a], embedding[b]
            used_edges_set.add((ha, hb))
            used_edges_set.add((hb, ha))
        result[idx] = embedding
        total_edges += len(tree_edges)
    
    # Verify and convert to proper int lists
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            # Ensure all are Python ints
            final_result.append([int(x) for x in result[i]])
    
    return final_result

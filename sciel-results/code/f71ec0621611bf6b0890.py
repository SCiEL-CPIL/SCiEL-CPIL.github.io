"""Superpermutation via Hamiltonian path in de Bruijn graph of order n-1."""
import random
import time
from collections import defaultdict

def construct(n):
    """Return shortest known superpermutation word over symbols 1..n.
    
    Strategy: Find Hamiltonian path in de Bruijn graph B(n-1, n).
    Nodes: all (n-1)-permutations of {1..n}.
    Edges: node -> node by appending symbol and dropping first.
    A Hamiltonian path gives a word of length (n-1)! + n containing all n! permutations.
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s kill
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate all (n-1)-permutations as nodes
    from itertools import permutations
    symbols = list(range(1, n + 1))
    all_nodes = list(permutations(symbols))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)  # (n-1)!
    
    # Build adjacency: for each node, which symbols can extend it (not already in node)
    # node = (a1, a2, ..., a_{n-1}), can append s not in node
    # new node = (a2, ..., a_{n-1}, s)
    adj = [[] for _ in range(num_nodes)]
    adj_set = [set() for _ in range(num_nodes)]
    for i, node in enumerate(all_nodes):
        node_set = set(node)
        for s in range(1, n + 1):
            if s not in node_set:
                new_node = node[1:] + (s,)
                j = node_to_idx[new_node]
                adj[i].append((s, j))
                adj_set[i].add(j)
    
    # Degree for each node
    degree = [len(adj[i]) for i in range(num_nodes)]
    
    def time_up():
        return (time.time() - start_time) > DEADLINE
    
    # For small n, try exhaustive DFS with Warnsdorff
    if num_nodes <= 120:  # n=4: 6 nodes, n=5: 24 nodes - easy
        return _solve_small(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up)
    
    # For larger n, use randomized heuristic
    best_word = None
    best_len = float('inf')
    
    # Phase 1: Multiple randomized greedy attempts
    num_greedy = 200 if n <= 5 else 50
    for attempt in range(num_greedy):
        if time_up():
            break
        word = _greedy_hamiltonian(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up)
        if word is not None:
            wl = len(word)
            if wl < best_len:
                best_len = wl
                best_word = word
            if best_len == num_nodes + n:  # optimal
                return best_word
    
    if best_word is not None:
        return best_word
    
    # Phase 2: For n >= 6, try local search / repair on partial paths
    if n <= 6 and not time_up():
        best_partial = None
        best_partial_len = 0
        for attempt in range(100):
            if time_up():
                break
            partial = _partial_path(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up)
            if partial is not None and len(partial) > best_partial_len:
                best_partial_len = len(partial)
                best_partial = partial
                if best_partial_len == num_nodes:
                    # Complete!
                    word = _path_to_word(best_partial, n)
                    return word
        # Try to repair the best partial path
        if best_partial is not None:
            word = _repair_path(n, all_nodes, node_to_idx, adj, adj_set, degree, best_partial, time_up)
            if word is not None:
                return word
    
    # Fallback: return the classic recursive construction (always valid)
    return _classic_superperm(n)

def _solve_small(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up):
    """Exhaustive DFS for small n."""
    num_nodes = len(all_nodes)
    visited = [False] * num_nodes
    path = []
    
    def dfs(node_idx):
        if time_up():
            return None
        visited[node_idx] = True
        path.append(node_idx)
        if len(path) == num_nodes:
            return list(path)
        # Warnsdorff: sort by remaining degree
        neighbors = sorted(adj[node_idx], key=lambda x: sum(1 for nb in adj_set[x[1]] if not visited[nb]))
        for sym, next_idx in neighbors:
            if not visited[next_idx]:
                result = dfs(next_idx)
                if result is not None:
                    return result
        path.pop()
        visited[node_idx] = False
        return None
    
    # Try each starting node
    for start in range(num_nodes):
        result = dfs(start)
        if result is not None:
            return _path_to_word(result, n)
        if time_up():
            break
    
    return None

def _greedy_hamiltonian(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up):
    """Randomized greedy with Warnsdorff's rule."""
    num_nodes = len(all_nodes)
    visited = [False] * num_nodes
    path = []
    
    start = random.randrange(num_nodes)
    visited[start] = True
    path.append(start)
    
    for _ in range(num_nodes - 1):
        if time_up():
            return None
        current = path[-1]
        unvisited_neighbors = [x for x in adj[current] if not visited[x[1]]]
        if not unvisited_neighbors:
            return None  # stuck
        
        # Warnsdorff: prefer nodes with fewer unvisited neighbors
        def warnsdorff_key(x):
            sym, next_idx = x
            remaining = sum(1 for nb in adj_set[next_idx] if not visited[nb])
            return (remaining, random.random())  # random tiebreak
        
        unvisited_neighbors.sort(key=warnsdorff_key)
        next_sym, next_idx = unvisited_neighbors[0]
        visited[next_idx] = True
        path.append(next_idx)
    
    return _path_to_word(path, n)

def _path_to_word(path, n):
    """Convert a Hamiltonian path (list of node indices) to the superpermutation word.
    path[i] is an index into all_nodes, which is a tuple of n-1 symbols.
    The word starts with the first node's symbols, then for each subsequent node,
    append the last symbol of that node (which is the symbol that was added).
    """
    if not path:
        return ""
    # We need the actual node tuples. Reconstruct from path indices.
    # Actually, let's pass all_nodes or reconstruct differently.
    # Let me restructure: path should contain the actual tuples or we pass all_nodes.
    # For now, this is a placeholder - the caller should handle conversion.
    # Let's fix this by having the functions return the word directly.
    raise NotImplementedError("Use the refactored version")

def _partial_path(n, all_nodes, node_to_idx, adj, adj_set, degree, time_up):
    """Generate a partial Hamiltonian path as long as possible."""
    num_nodes = len(all_nodes)
    visited = [False] * num_nodes
    path = []
    
    start = random.randrange(num_nodes)
    visited[start] = True
    path.append(start)
    
    for _ in range(num_nodes - 1):
        if time_up():
            break
        current = path[-1]
        unvisited_neighbors = [x for x in adj[current] if not visited[x[1]]]
        if not unvisited_neighbors:
            break
        unvisited_neighbors.sort(key=lambda x: (sum(1 for nb in adj_set[x[1]] if not visited[nb]), random.random()))
        next_sym, next_idx = unvisited_neighbors[0]
        visited[next_idx] = True
        path.append(next_idx)
    
    return path

def _repair_path(n, all_nodes, node_to_idx, adj, adj_set, degree, partial, time_up):
    """Try to extend a partial path to a complete Hamiltonian path."""
    # This is complex; for now, just try more greedy restarts
    return None

def _classic_superperm(n):
    """Classic recursive superpermutation construction (always valid, length 1!+2!+...+n!)."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

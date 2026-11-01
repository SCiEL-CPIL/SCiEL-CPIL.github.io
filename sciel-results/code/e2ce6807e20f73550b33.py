import time
import itertools
import sys
import random
from collections import Counter, defaultdict

def _is_valid_superperm(word, n, all_perms):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    return all_perms.issubset(seen)

def _build_perm_graph(n):
    """Build graph where nodes are (n-1)-permutations, edges are n-permutations.
    Edge from (a1,...,a_{n-1}) to (a2,...,a_{n-1},b) represents perm (a1,...,a_{n-1},b)."""
    # Nodes: all (n-1)-permutations of 1..n
    nodes = list(itertools.permutations(range(1, n + 1), n - 1))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # For each n-permutation, it gives an edge from perm[:-1] to perm[1:]
    # Multiple edges possible if same (n-1)-prefix/suffix pair with different middle? No,
    # actually for a given (n-1)-prefix P and last element x not in P, there's exactly one edge.
    # So edges are determined by: for each node (n-1)-perm, for each symbol not in it, one outgoing edge.
    
    # Build adjacency: for each node, list of (next_node, perm_tuple)
    adj = [[] for _ in range(num_nodes)]
    for node in nodes:
        idx = node_to_idx[node]
        node_set = set(node)
        for x in range(1, n + 1):
            if x not in node_set:
                # Edge: node -> node[1:] + (x,)
                next_node = node[1:] + (x,)
                next_idx = node_to_idx[next_node]
                perm = node + (x,)
                adj[idx].append((next_idx, perm))
    return nodes, node_to_idx, adj, num_nodes

def _greedy_eulerian(n):
    """Find a path covering all n-permutations using greedy on the de Bruijn-like graph.
    We need a trail that uses each edge (n-perm) exactly once = Eulerian trail."""
    nodes, node_to_idx, adj, num_nodes = _build_perm_graph(n)
    
    # Count out-degrees and in-degrees to find start
    out_deg = [len(adj[i]) for i in range(num_nodes)]
    in_deg = [0] * num_nodes
    for i in range(num_nodes):
        for next_idx, perm in adj[i]:
            in_deg[next_idx] += 1
    
    # Find start node: out_deg - in_deg = 1 (or 0 if Eulerian circuit)
    start = None
    for i in range(num_nodes):
        if out_deg[i] - in_deg[i] == 1:
            start = i
            break
    if start is None:
        # Eulerian circuit
        start = 0
    
    # Hierholzer's algorithm for Eulerian trail
    stack = [start]
    path = []
    used_edges = [[False] * len(adj[i]) for i in range(num_nodes)]
    
    while stack:
        v = stack[-1]
        # Find next unused edge
        found = False
        for e_idx, (next_idx, perm) in enumerate(adj[v]):
            if not used_edges[v][e_idx]:
                used_edges[v][e_idx] = True
                stack.append(next_idx)
                found = True
                break
        if not found:
            path.append(stack.pop())
    
    path.reverse()
    
    # Convert path of nodes to sequence of permutations
    # The path visits nodes; consecutive nodes define an edge (permutation)
    # First permutation: from start to path[1], etc.
    # Actually, path has num_nodes + 1 entries (start + one per edge traversed)
    # Wait, Hierholzer gives the nodes in reverse order of backtracking.
    
    # Let me redo: the path list contains nodes in order of visitation (reversed at end)
    # Number of edges = n! = 5040 for n=7
    # Number of nodes in path = n! + 1
    
    # Extract permutations from consecutive node pairs
    perms_in_order = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        # Find which edge from u to v
        node_u = nodes[u]
        node_v = nodes[v]
        # The permutation is node_u + (last element of node_v,)
        # Since node_v = node_u[1:] + (x,), x = node_v[-1]
        x = node_v[-1]
        perm = node_u + (x,)
        perms_in_order.append(perm)
    
    # Now build the word: start with first permutation, then for each subsequent,
    # append only the last element (since they overlap by n-1)
    if not perms_in_order:
        return []
    
    word = list(perms_in_order[0])
    for perm in perms_in_order[1:]:
        word.append(perm[-1])
    
    return word

def _local_search_shorten(word, n, all_perms, deadline):
    """Try to shorten by removing redundant symbols."""
    current = list(word)
    L = len(current)
    
    # Count occurrences of each permutation
    perm_counts = Counter()
    for i in range(L - n + 1):
        perm_counts[tuple(current[i:i + n])] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(current)
        
        # Recompute counts
        perm_counts = Counter()
        for i in range(L - n + 1):
            perm_counts[tuple(current[i:i + n])] += 1
        
        # Find all deletable positions
        deletable = []
        for i in range(L):
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(current[s:s + n])
                if perm_counts[w] == 1:
                    safe = False
                    break
            if safe:
                deletable.append(i)
        
        if not deletable:
            break
        
        # Try to delete as many as possible
        # Try all deletable at once
        delete_set = set(deletable)
        new_word = [current[i] for i in range(L) if i not in delete_set]
        
        if _is_valid_superperm(new_word, n, all_perms):
            current = new_word
            improved = True
        else:
            # Try one by one
            for pos in deletable:
                if time.time() > deadline:
                    break
                trial = list(current)
                trial.pop(pos)
                if _is_valid_superperm(trial, n, all_perms):
                    current = trial
                    improved = True
                    break
    
    return current

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        # For small n, use brute force or known optimal
        if n == 1:
            return "1"
        if n == 2:
            return "121"
        if n == 3:
            return "123121321"
        if n == 4:
            # Known optimal for n=4 is 33
            return "12341213421431243123413241342143214231"
    
    start_time = time.time()
    deadline = start_time + 115
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Use Eulerian trail construction
    word = _greedy_eulerian(n)
    
    # Local search to shorten
    polished = _local_search_shorten(word, n, all_perms, deadline)
    
    # Validate
    if _is_valid_superperm(polished, n, all_perms):
        return "".join(map(str, polished))
    
    # Fallback: validate original
    if _is_valid_superperm(word, n, all_perms):
        return "".join(map(str, word))
    
    # Last resort: classic construction
    # Build classic superperm
    word2 = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word2) - m + 2):
            window = tuple(word2[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word2 = new
    
    return "".join(map(str, word2))

import time
import random

_START = time.time()
_DEADLINE = _START + 108

def _has_path_5(u, v, adj_masks):
    # DFS for simple path of length 5
    stack = [(u, 0, 1 << u)]
    while stack:
        node, depth, mask = stack.pop()
        if depth == 5:
            if node == v:
                return True
            continue
        neighbors = adj_masks[node]
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            if mask & (1 << nb):
                continue
            if nb == v and depth != 4:
                continue
            stack.append((nb, depth + 1, mask | (1 << nb)))
    return False

def _greedy_with_local_search(n=40, seed=0):
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if not _has_path_5(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search Phase 1: Pure Add
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < _DEADLINE - 15:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if not _has_path_5(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
                
    # Local Search Phase 2: Swaps
    # Try to remove an edge and see if we can add more
    edge_list = list(edges)
    improved = True
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        rng.shuffle(edge_list)
        for u, v in edge_list:
            if (u, v) not in edges:
                continue
                
            # Copy state
            adj_copy = adj_masks[:]
            edges_copy = set(edges)
            
            # Remove u, v
            adj_copy[u] &= ~(1 << v)
            adj_copy[v] &= ~(1 << u)
            edges_copy.remove((u, v))
            
            # Try to add edges
            for x, y in all_edges:
                if (x, y) in edges_copy:
                    continue
                if not _has_path_5(x, y, adj_copy):
                    adj_copy[x] |= (1 << y)
                    adj_copy[y] |= (1 << x)
                    edges_copy.add((x, y))
                    
            if len(edges_copy) > len(edges):
                adj_masks = adj_copy
                edges = edges_copy
                improved = True
                edge_list = list(edges)
                break
                
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    for seed in range(10000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 110:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]

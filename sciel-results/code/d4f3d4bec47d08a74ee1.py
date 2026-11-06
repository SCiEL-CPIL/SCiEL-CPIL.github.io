import time
import sys
from collections import deque

_START = time.time()
DEADLINE = _START + 130.0

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_longest_cycle_from(start, rem, n, max_len=None):
    """
    Find a long cycle starting from `start` using a modified DFS that
    tries to find the longest simple cycle.
    """
    best = None
    best_len = 0
    
    # DFS with path tracking
    # We want to find a cycle that returns to start
    # Use a stack: (node, iterator, path_so_far)
    # But to find longest, we need backtracking
    
    # For efficiency, we'll do a DFS that explores all paths up to a certain length
    # and records the longest cycle found
    
    sys.setrecursionlimit(10000)
    
    def dfs(u, visited, parent, path):
        nonlocal best, best_len
        if time.time() > DEADLINE:
            return
        for v in list(rem[u]):
            if v == parent:
                continue
            if v == start and len(path) >= 2:
                # Found a cycle: path + [start]
                cycle = path + [start]
                # Actually the cycle is path[0]...path[-1]...back to start
                # path is [start, ..., u], and edge u-start exists
                # So cycle is path (which starts with start) and we close it
                # But we want the cycle without repeating start at end
                cyc = list(path)
                if len(cyc) > best_len:
                    best_len = len(cyc)
                    best = cyc
            elif v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, visited, u, path)
                path.pop()
                visited.remove(v)
    
    visited = {start}
    path = [start]
    dfs(start, visited, -1, path)
    
    return best


def _find_any_cycle(rem, n, start_hint=None):
    """
    Find any cycle in the remaining graph using DFS.
    Returns the cycle as a list of vertices, or None.
    """
    for start in range(n):
        if not rem[start]:
            continue
        if start_hint is not None and start != start_hint:
            continue
        parent = {start: -1}
        visited = {start}
        # Iterative DFS
        stack = [(start, iter(sorted(rem[start])))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v == parent.get(u, -1):
                    continue
                if v in visited:
                    # Found a cycle
                    cyc = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    return cyc
                parent[v] = u
                visited.add(v)
                stack.append((v, iter(sorted(rem[v]))))
                advanced = True
                break
            if not advanced:
                stack.pop()
        if start_hint is not None:
            break
    
    return None


def _decompose(edges, n):
    """
    Decompose edge set into edge-disjoint cycles + leftover single edges,
    minimizing piece count.
    """
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)
    
    pieces = []
    
    # Strategy: repeatedly find the longest cycle possible, remove it, repeat.
    # To find longer cycles, we try from high-degree vertices and use a
    # longest-cycle DFS.
    
    while time.time() < DEADLINE:
        # Check if any edges remain
        total_edges = sum(len(rem[u]) for u in range(n)) // 2
        if total_edges == 0:
            break
        
        # Try to find a long cycle
        # Sort vertices by degree (descending)
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)
        
        best_cycle = None
        best_len = 0
        
        # Try from a few high-degree starts
        for deg, start in degs[:8]:
            if time.time() > DEADLINE:
                break
            cyc = _find_longest_cycle_from(start, rem, n)
            if cyc is not None and len(cyc) > best_len:
                best_len = len(cyc)
                best_cycle = cyc
            # If we found a cycle of length >= 5, it's pretty good
            if best_len >= 6:
                break
        
        # If no long cycle found, try to find any cycle
        if best_cycle is None:
            # Try from highest degree vertex
            start = degs[0][1] if degs else 0
            best_cycle = _find_any_cycle(rem, n, start_hint=start)
        
        if best_cycle is None:
            # No cycle found, break
            break
        
        # Remove cycle edges
        cyc = best_cycle
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(cyc)
    
    # Leftover edges
    leftover = set()
    for u in range(n):
        for v in rem[u]:
            if u < v:
                leftover.add((u, v))
    for u, v in sorted(leftover):
        pieces.append([u, v])
    
    return pieces


def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results

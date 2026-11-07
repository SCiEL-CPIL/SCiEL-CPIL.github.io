"""Improved program using constraint-based approach for edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']

# Build adjacency and edge set
ADJ = [[] for _ in range(N)]
EDGE_SET = set()
for u, v in P['edges']:
    ADJ[u].append(v)
    ADJ[v].append(u)
    if u > v:
        u, v = v, u
    EDGE_SET.add((u, v))

# Sort neighbors by degree (ascending) for better pruning
for i in range(N):
    ADJ[i].sort(key=lambda x: len(ADJ[x]))

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_hamiltonian_cycles(deadline, max_cycles=200):
    """Find Hamiltonian cycles using backtracking with pruning."""
    cycles = []
    
    def dfs(path, visited, used_edges):
        if time.time() > deadline:
            return False
        if len(cycles) >= max_cycles:
            return True
        
        if len(path) == N:
            # Check if last connects to first
            last, first = path[-1], path[0]
            e = _edge_key(last, first)
            if e in EDGE_SET and e not in used_edges:
                cycles.append(list(path))
            return len(cycles) >= max_cycles
        
        cur = path[-1]
        for nxt in ADJ[cur]:
            if nxt not in visited:
                e = _edge_key(cur, nxt)
                if e not in used_edges:
                    path.append(nxt)
                    visited.add(nxt)
                    used_edges.add(e)
                    
                    # Pruning: check if remaining vertices can still be connected
                    remaining = [v for v in range(N) if v not in visited]
                    if remaining:
                        # Check if any remaining vertex has no available edges to unvisited or current
                        for v in remaining:
                            has_edge = False
                            for w in ADJ[v]:
                                if w not in visited:
                                    if _edge_key(v, w) not in used_edges:
                                        has_edge = True
                                        break
                            if not has_edge:
                                break
                        else:
                            if dfs(path, visited, used_edges):
                                return True
                    
                    path.pop()
                    visited.remove(nxt)
                    used_edges.remove(e)
        
        return len(cycles) >= max_cycles
    
    # Start from vertex 0 (fix to reduce symmetry)
    dfs([0], {0}, set())
    return cycles

def _cycles_conflict(c1, c2):
    """Check if two cycles share any edge."""
    for i in range(len(c1)):
        u, v = c1[i], c1[(i + 1) % len(c1)]
        e = _edge_key(u, v)
        for j in range(len(c2)):
            u2, v2 = c2[j], c2[(j + 1) % len(c2)]
            if _edge_key(u2, v2) == e:
                return True
    return False

def _max_independent_set(conflict_matrix, n):
    """Find maximum independent set using branch and bound."""
    best = 0
    best_set = []
    
    def branch(idx, current, available):
        nonlocal best, best_set
        if idx >= n:
            if len(current) > best:
                best = len(current)
                best_set = list(current)
            return
        
        # Upper bound
        if len(current) + len(available) <= best:
            return
        
        # Option 1: include vertex idx
        if idx in available:
            new_current = current + [idx]
            new_available = set()
            for v in available:
                if v != idx and not conflict_matrix[idx][v]:
                    new_available.add(v)
            branch(idx + 1, new_current, new_available)
        
        # Option 2: exclude vertex idx
        branch(idx + 1, current, available - {idx})
    
    branch(0, [], set(range(n)))
    return best_set

def construct():
    """Find maximum number of edge-disjoint Hamiltonian cycles."""
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    # Phase 1: Find Hamiltonian cycles
    random.seed(42)
    cycles = _find_hamiltonian_cycles(deadline, max_cycles=100)
    
    if not cycles:
        # Fallback: try with different starting vertex
        for start in range(1, N):
            if time.time() > deadline:
                break
            # Temporarily modify to start from different vertex
            # For simplicity, just return what we have
            break
    
    # Phase 2: Find maximum set of edge-disjoint cycles
    n_cycles = len(cycles)
    if n_cycles <= 1:
        return [[int(v) for v in cycles[0]]] if cycles else [list(range(N))]
    
    # Build conflict matrix
    conflict = [[False] * n_cycles for _ in range(n_cycles)]
    for i in range(n_cycles):
        for j in range(i + 1, n_cycles):
            if _cycles_conflict(cycles[i], cycles[j]):
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Find maximum independent set
    if time.time() < deadline - 5:
        independent_set = _max_independent_set(conflict, n_cycles)
    else:
        # Fallback: greedy
        independent_set = []
        used = set()
        for i in range(n_cycles):
            if i in used:
                continue
            can_add = True
            for j in independent_set:
                if conflict[i][j]:
                    can_add = False
                    break
            if can_add:
                independent_set.append(i)
                for j in range(n_cycles):
                    if conflict[i][j]:
                        used.add(j)
    
    # Build result
    result = []
    for idx in independent_set:
        result.append([int(v) for v in cycles[idx]])
    
    if not result:
        result = [[int(v) for v in cycles[0]]] if cycles else [list(range(N))]
    
    return result[:10]

# EVOLVE-BLOCK-END

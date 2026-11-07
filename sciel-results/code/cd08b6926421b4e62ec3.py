"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = frozenset(range(N))
EDGE_SET = set(frozenset(e) for e in P['edges'])
ADJ = defaultdict(set)
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

EDGE_LOOKUP = set()
for u, v in P['edges']:
    EDGE_LOOKUP.add((u, v))
    EDGE_LOOKUP.add((v, u))


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _is_valid_cycle(cycle, used_edges):
    """Check if cycle is valid: distinct vertices, all edges exist and unused."""
    if len(cycle) < 3:
        return False
    verts = set(cycle)
    if len(verts) != len(cycle):
        return False
    if verts != ALL_VERTS:
        return False
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        ek = _edge_key(u, v)
        if ek not in EDGE_LOOKUP:
            return False
        if ek in used_edges:
            return False
    return True


def _find_hamiltonian_cycle(used_edges, start=0, deadline=None, max_steps=500000, shuffle_neighbors=False):
    """Find a Hamiltonian cycle avoiding used_edges using iterative DFS with heuristics."""
    n = N
    
    # Precompute available neighbors with degree-based heuristic
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    if shuffle_neighbors:
        for v in range(n):
            random.shuffle(avail[v])
    else:
        # Sort by degree (fewest options first - most constrained)
        for v in range(n):
            avail[v].sort(key=lambda u: len(avail[u]))
    
    path = [start]
    visited = {start}
    steps = [0]
    
    # Iterative DFS with explicit stack to avoid recursion limits
    stack = [(start, iter(avail[start]))]
    
    while stack:
        if deadline is not None and time.time() > deadline:
            return None
        steps[0] += 1
        if steps[0] > max_steps:
            return None
        
        current, neighbors = stack[-1]
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            # Dead end at full length
            path.pop()
            visited.discard(current)
            stack.pop()
            continue
        
        found_next = False
        for nxt in neighbors:
            if nxt in visited:
                continue
            if nxt == start:
                continue
            
            visited.add(nxt)
            path.append(nxt)
            
            # Check if we can continue from nxt
            has_more = False
            if len(path) < n:
                for nnxt in avail[nxt]:
                    if nnxt not in visited and nnxt != start:
                        has_more = True
                        break
            else:
                # At last position, check if we can close
                if _edge_key(nxt, start) not in used_edges and (nxt, start) in EDGE_LOOKUP:
                    return list(path)
            
            if has_more or len(path) == n:
                stack.append((nxt, iter(avail[nxt])))
                found_next = True
                break
        
        if not found_next:
            # Backtrack
            path.pop()
            visited.discard(current)
            stack.pop()
    
    return None


def _try_random_start(used_edges, deadline, max_steps=150000):
    """Try finding Hamiltonian cycle with random neighbor ordering."""
    n = N
    start = random.randint(0, n - 1)
    
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    # Randomize neighbor order
    for v in range(n):
        random.shuffle(avail[v])
    
    path = [start]
    visited = {start}
    steps = [0]
    
    # Iterative DFS
    stack = [(start, iter(avail[start]))]
    
    while stack:
        if deadline is not None and time.time() > deadline:
            return None
        steps[0] += 1
        if steps[0] > max_steps:
            return None
        
        current, neighbors = stack[-1]
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            path.pop()
            visited.discard(current)
            stack.pop()
            continue
        
        found_next = False
        for nxt in neighbors:
            if nxt in visited:
                continue
            if nxt == start:
                continue
            
            visited.add(nxt)
            path.append(nxt)
            
            has_more = False
            if len(path) < n:
                for nnxt in avail[nxt]:
                    if nnxt not in visited and nnxt != start:
                        has_more = True
                        break
            else:
                if _edge_key(nxt, start) not in used_edges and (nxt, start) in EDGE_LOOKUP:
                    return list(path)
            
            if has_more or len(path) == n:
                stack.append((nxt, iter(avail[nxt])))
                found_next = True
                break
        
        if not found_next:
            path.pop()
            visited.discard(current)
            stack.pop()
    
    return None


def _smart_edge_removal(used_edges, cycles):
    """
    Smart strategy: after finding cycles, analyze which edges are most critical
    and try alternative orderings.
    """
    # This is a placeholder for future enhancement
    return cycles


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Fallback: the known Hamiltonian cycle
    fallback_cycle = list(range(N))
    
    cycles = []
    used_edges = set()
    
    best_cycles = []
    
    # Phase 1: Deterministic first cycle
    deadline1 = min(deadline, start_time + 10)
    cycle = _find_hamiltonian_cycle(used_edges, start=0, deadline=deadline1, max_steps=300000)
    if cycle is not None:
        best_cycles.append(cycle)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
    
    # If we didn't find one, use the fallback
    if not best_cycles:
        best_cycles.append(fallback_cycle)
        for i in range(len(fallback_cycle)):
            u = fallback_cycle[i]
            v = fallback_cycle[(i + 1) % len(fallback_cycle)]
            used_edges.add(_edge_key(u, v))
    
    # Phase 2: Random restarts to find more cycles
    random.seed(42)
    
    # Try multiple random seeds for diversity
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() >= deadline - 3:
            break
        if len(best_cycles) >= 10:
            break
        
        random.seed(seed)
        
        for attempt in range(50):
            if time.time() >= deadline - 2:
                break
            if len(best_cycles) >= 10:
                break
            
            deadline_attempt = min(deadline - 1, time.time() + 5)
            cycle = _try_random_start(used_edges, deadline=deadline_attempt, max_steps=80000)
            
            if cycle is not None:
                if _is_valid_cycle(cycle, used_edges):
                    best_cycles.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add(_edge_key(u, v))
                    # If we found a cycle, try a few more with same seed
                    break
    
    # Phase 3: Polish pass - try harder with different heuristics
    if len(best_cycles) < 10 and time.time() < deadline - 5:
        # Try with different start vertices
        for start_v in range(N):
            if time.time() >= deadline - 3:
                break
            if len(best_cycles) >= 10:
                break
            
            deadline_attempt = min(deadline - 2, time.time() + 4)
            cycle = _find_hamiltonian_cycle(used_edges, start=start_v, deadline=deadline_attempt, max_steps=100000, shuffle_neighbors=True)
            
            if cycle is not None:
                if _is_valid_cycle(cycle, used_edges):
                    best_cycles.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add(_edge_key(u, v))
    
    # Phase 4: Final aggressive randomization
    if len(best_cycles) < 10 and time.time() < deadline - 2:
        for _ in range(100):
            if time.time() >= deadline - 1:
                break
            if len(best_cycles) >= 10:
                break
            
            deadline_attempt = min(deadline - 0.5, time.time() + 2)
            cycle = _try_random_start(used_edges, deadline=deadline_attempt, max_steps=50000)
            
            if cycle is not None:
                if _is_valid_cycle(cycle, used_edges):
                    best_cycles.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add(_edge_key(u, v))
    
    # Return the cycles as lists of ints
    result = [list(c) for c in best_cycles[:10]]
    
    # Ensure types are correct
    for i, cycle in enumerate(result):
        result[i] = [int(v) for v in cycle]
    
    return result


# EVOLVE-BLOCK-END

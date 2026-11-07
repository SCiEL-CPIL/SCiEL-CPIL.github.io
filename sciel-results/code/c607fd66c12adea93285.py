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


def _get_avail(used_edges):
    """Get available neighbors for each vertex given used edges."""
    avail = {}
    for v in range(N):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    return avail


def _find_ham_iterative(used_edges, start, deadline, max_steps=2000000, seed=None):
    """Iterative DFS to find Hamiltonian cycle avoiding used_edges."""
    if seed is not None:
        random.seed(seed)
    
    avail = _get_avail(used_edges)
    
    # Sort by degree heuristic (fewest options first)
    for v in range(N):
        avail[v].sort(key=lambda u: len(avail[u]))
    
    # Shuffle if we want randomness
    if seed is not None:
        for v in range(N):
            random.shuffle(avail[v])
        # Re-sort with slight randomization
        for v in range(N):
            avail[v].sort(key=lambda u: (len(avail[u]), random.random()))
    
    n = N
    path = [start]
    visited = {start}
    steps = 0
    
    # Iterative DFS with explicit stack
    # Stack entries: (node, neighbor_index)
    stack = [(start, 0)]
    
    while stack:
        if deadline is not None and time.time() > deadline:
            return None
        steps += 1
        if steps > max_steps:
            return None
        
        node, ni = stack[-1]
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            # Dead end at full length
            path.pop()
            visited.discard(node)
            stack.pop()
            if stack:
                prev_node, prev_ni = stack[-1]
                stack[-1] = (prev_node, prev_ni + 1)
            continue
        
        found_next = False
        while ni < len(avail[node]):
            nxt = avail[node][ni]
            ni += 1
            if nxt in visited:
                continue
            if nxt == start:
                continue
            # Try this neighbor
            stack[-1] = (node, ni)
            path.append(nxt)
            visited.add(nxt)
            stack.append((nxt, 0))
            found_next = True
            break
        
        if not found_next:
            # Backtrack
            path.pop()
            visited.discard(node)
            stack.pop()
            if stack:
                prev_node, prev_ni = stack[-1]
                stack[-1] = (prev_node, prev_ni + 1)
    
    return None


def _try_find_cycles(used_edges, deadline, max_cycles=10, num_restarts=500):
    """Try to find multiple edge-disjoint Hamiltonian cycles."""
    cycles = []
    
    for attempt in range(num_restarts):
        if deadline is not None and time.time() >= deadline - 2:
            break
        if len(cycles) >= max_cycles:
            break
        
        # Vary the start vertex and seed
        start = attempt % N
        seed = attempt * 7919 + 12345
        
        cycle = _find_ham_iterative(used_edges, start, deadline=deadline, 
                                     max_steps=500000, seed=seed)
        
        if cycle is not None:
            if _is_valid_cycle(cycle, used_edges):
                cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add(_edge_key(u, v))
    
    return cycles


def _swap_to_improve(cycles, deadline):
    """Try to swap cycles to free up edges and find more."""
    if not cycles:
        return cycles
    
    best = list(cycles)
    used_edges = set()
    for cycle in best:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
    
    # Try removing each cycle and re-optimizing
    for idx in range(len(best)):
        if deadline is not None and time.time() >= deadline - 5:
            break
        
        removed = best[idx]
        # Remove its edges
        for i in range(len(removed)):
            u = removed[i]
            v = removed[(i + 1) % len(removed)]
            used_edges.discard(_edge_key(u, v))
        
        remaining = best[:idx] + best[idx+1:]
        
        # Try to find more cycles with the freed edges
        new_cycles = _try_find_cycles(used_edges, deadline, max_cycles=10-len(remaining), 
                                       num_restarts=50)
        
        # See if we can get more total cycles
        if len(remaining) + len(new_cycles) > len(best):
            best = remaining + new_cycles
            # Rebuild used_edges
            used_edges = set()
            for cycle in best:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add(_edge_key(u, v))
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Known fallback Hamiltonian cycle
    fallback_cycle = list(range(N))
    
    used_edges = set()
    cycles = []
    
    # First, try deterministic approach with sorted neighbors
    deadline1 = min(deadline, start_time + 10)
    cycle = _find_ham_iterative(used_edges, 0, deadline=deadline1, max_steps=1000000, seed=None)
    if cycle is not None:
        cycles.append(cycle)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
    
    if not cycles:
        cycles.append(fallback_cycle)
        for i in range(len(fallback_cycle)):
            u = fallback_cycle[i]
            v = fallback_cycle[(i + 1) % len(fallback_cycle)]
            used_edges.add(_edge_key(u, v))
    
    # Continue finding more cycles with random restarts
    more_cycles = _try_find_cycles(used_edges, deadline, max_cycles=10, num_restarts=300)
    cycles.extend(more_cycles)
    
    # Try to improve by swapping
    if time.time() < deadline - 10:
        cycles = _swap_to_improve(cycles, deadline)
    
    # Final verification and cleanup
    result = []
    final_used_edges = set()
    for cycle in cycles[:10]:
        if _is_valid_cycle(cycle, final_used_edges):
            result.append([int(v) for v in cycle])
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                final_used_edges.add(_edge_key(u, v))
    
    if not result:
        # Absolute fallback
        result = [[int(v) for v in fallback_cycle]]
    
    return result


# EVOLVE-BLOCK-END

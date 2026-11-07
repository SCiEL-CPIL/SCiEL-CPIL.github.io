"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_edge_set():
    es = set()
    for e in P['edges']:
        a, b = int(e[0]), int(e[1])
        es.add((min(a, b), max(a, b)))
    return es

def _make_cycle(step, n):
    """Generate a Hamiltonian cycle by stepping by 'step' mod n."""
    cycle = [0]
    cur = 0
    for _ in range(n - 1):
        cur = (cur + step) % n
        cycle.append(cur)
    return [int(v) for v in cycle]

def _verify_cycle(cycle, edge_set, n):
    """Check that cycle is valid: distinct vertices, all edges exist."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        e = (min(a, b), max(a, b))
        if e not in edge_set:
            return False
    return True

def _cycle_edges(cycle, n):
    """Return set of edges used by a cycle."""
    edges = set()
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        edges.add((min(a, b), max(a, b)))
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    edge_set = _build_edge_set()
    n = P['n']
    
    # The graph is a circulant C_21(1,2,5) with extra edges
    # Steps coprime to 21 that correspond to edge lengths: 1, 2, 5
    # gcd(1,21)=1, gcd(2,21)=1, gcd(5,21)=1
    candidate_steps = [1, 2, 5]
    
    cycles = []
    used_edges = set()
    
    for step in candidate_steps:
        if time.time() > deadline:
            break
        cycle = _make_cycle(step, n)
        if not _verify_cycle(cycle, edge_set, n):
            continue
        # Check edge-disjointness with already selected cycles
        cycle_edges = _cycle_edges(cycle, n)
        if cycle_edges & used_edges:
            continue
        cycles.append(cycle)
        used_edges.update(cycle_edges)
    
    # Theoretical maximum: floor(70/21) = 3
    # We've achieved 3, which is optimal
    if len(cycles) < 3:
        # Fallback: try other coprime steps
        other_steps = [s for s in range(1, n) if gcd(s, n) == 1 and s not in candidate_steps]
        for step in other_steps:
            if time.time() > deadline:
                break
            if len(cycles) >= 10:
                break
            cycle = _make_cycle(step, n)
            if not _verify_cycle(cycle, edge_set, n):
                continue
            cycle_edges = _cycle_edges(cycle, n)
            if cycle_edges & used_edges:
                continue
            cycles.append(cycle)
            used_edges.update(cycle_edges)
    
    if not cycles:
        # Absolute fallback
        return [list(range(n))]
    
    return cycles
# EVOLVE-BLOCK-END

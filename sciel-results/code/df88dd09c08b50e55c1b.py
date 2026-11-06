import time
import random

N = 40
DEADLINE = 85  # seconds, safety margin before 90s kill

def _add_edge(edges, u, v):
    if u > v:
        u, v = v, u
    edges.add((u, v))

def _validate_result(edges_set, cycles):
    """Check all cycle edges exist in the graph and cycles are valid."""
    for L, cyc in enumerate(cycles, start=3):
        if len(cyc) != L:
            return False
        if len(set(cyc)) != L:
            return False
        for i in range(L):
            u, v = cyc[i], cyc[(i+1) % L]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edges_set:
                return False
    return True

def construct():
    start = time.time()
    rng = random.Random(42)
    
    # Initial solution: Hamiltonian cycle + 19 chords from vertex 0
    edges = set()
    for i in range(N):
        _add_edge(edges, i, (i + 1) % N)
    for k in range(2, 21):
        _add_edge(edges, 0, k)
    
    # Generate cycles for the initial solution
    def _make_cycles(edge_set):
        """Try to find pancyclic cycles using the current edge set."""
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cyc = list(range(N))
            elif L <= 21:
                k = L - 1
                # Check if chord (0, k) exists
                a, b = (0, k) if 0 < k else (k, 0)
                if (a, b) in edge_set:
                    cyc = list(range(L))
                else:
                    return None
            else:
                k = N - L + 1
                a, b = (0, k) if 0 < k else (k, 0)
                if (a, b) in edge_set:
                    cyc = [0] + list(range(N - 1, k - 1, -1))
                else:
                    return None
            cycles.append(cyc)
        return cycles
    
    best_edges = set(edges)
    best_cycles = _make_cycles(best_edges)
    
    # Try to reduce edge count by removing non-essential chords
    # We need to maintain pancyclicity. Start with all 19 chords and try removing them.
    
    def _try_remove_chord(edge_set, chord):
        """Try removing a chord and check if graph remains pancyclic."""
        e, v = chord
        new_edges = set(edge_set)
        new_edges.discard((e, v))
        cycles = _make_cycles(new_edges)
        if cycles is not None:
            return True, new_edges, cycles
        return False, None, None
    
    # Greedy removal: try removing each chord, keep removal if valid
    improved = True
    while improved:
        if time.time() - start > DEADLINE:
            break
        improved = False
        chords_to_try = [(0, k) for k in range(2, 21) if (0, k) in edges]
        rng.shuffle(chords_to_try)
        for chord in chords_to_try:
            if time.time() - start > DEADLINE:
                break
            success, new_edges, new_cycles = _try_remove_chord(edges, chord)
            if success:
                edges = new_edges
                best_edges = new_edges
                best_cycles = new_cycles
                improved = True
    
    # Try adding chords elsewhere to allow removal of multiple fan chords
    # This is a local search: try swapping one fan chord for a different chord
    
    all_vertices = list(range(N))
    for _ in range(100):  # bounded iterations
        if time.time() - start > DEADLINE:
            break
        # Try a random swap: remove one fan chord, add a random other chord
        fan_chords = [(0, k) for k in range(2, 21) if (0, k) in edges]
        if not fan_chords:
            break
        removed = rng.choice(fan_chords)
        
        # Generate random new chord (not already present, not a cycle edge)
        attempts = 0
        while attempts < 50:
            u = rng.choice(all_vertices)
            v = rng.choice(all_vertices)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            # Skip cycle edges
            if a + 1 == b and b - a < N:
                continue
            if b == 0 and a == N - 1:
                continue
            if (a, b) in edges:
                continue
            break
            attempts += 1
        else:
            continue
        attempts = 0
        new_chord = (a, b)
        
        new_edges = set(edges)
        new_edges.discard(removed)
        new_edges.add(new_chord)
        cycles = _make_cycles(new_edges)
        if cycles is not None and len(new_edges) < len(edges):
            edges = new_edges
            best_edges = new_edges
            best_cycles = cycles
    
    # Final validation
    if best_cycles is None:
        best_cycles = _make_cycles(best_edges)
    
    if best_cycles is None:
        # Fallback to original
        best_edges = set()
        for i in range(N):
            _add_edge(best_edges, i, (i + 1) % N)
        for k in range(2, 21):
            _add_edge(best_edges, 0, k)
        best_cycles = _make_cycles(best_edges)
    
    return {
        'edges': [tuple(sorted(e)) for e in best_edges],
        'cycles': best_cycles
    }

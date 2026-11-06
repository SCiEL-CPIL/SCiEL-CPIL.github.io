"""Sparse pancyclic graphs on 40 vertices.

Uses a base cycle plus carefully selected chords to cover all cycle lengths 3-40.
The key optimization is using a multi-chord cycle for length 21 to save one chord.
"""

N = 40

def construct():
    # Base cycle: 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Define chords explicitly for optimal coverage
    # Chords (0, d) for d in {3, 4, ..., 18} cover lengths {4,38}, {5,37}, ..., {19,23}
    # Chord (0, 19) covers lengths 20 and 22
    # Chord (0, 38) covers lengths 3 and 39
    # Length 21 is covered by multi-chord cycle: 0-19-20-...-38-0
    
    chords = []
    for d in range(3, 19):  # d = 3 to 18
        chords.append((0, d))
    chords.append((0, 19))  # covers 20 and 22
    chords.append((0, 38))  # covers 3 and 39
    
    # Build edge set
    edges = set(base_edges)
    for i, j in chords:
        edges.add((min(i, j), max(i, j)))
    
    # Construct cycles for each length 3..40
    cycles = []
    
    # Map length to chord for single-chord cycles
    length_to_chord = {}
    for i, j in chords:
        d = (j - i) % N
        len1 = d + 1
        len2 = N - d + 1
        if 3 <= len1 <= 39:
            length_to_chord.setdefault(len1, (i, j))
        if 3 <= len2 <= 39:
            length_to_chord.setdefault(len2, (i, j))
    
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle: the base cycle
            cycle = list(range(N))
        elif k == 21:
            # Special multi-chord cycle: 0-19-20-...-38-0
            cycle = [0, 19] + list(range(20, 39))
        else:
            # Single-chord cycle
            chord = length_to_chord.get(k)
            i, j = chord
            d = (j - i) % N
            len1 = d + 1
            len2 = N - d + 1
            
            if k == len1:
                # Cycle goes i -> i+1 -> ... -> j -> i
                cycle = []
                v = i
                for _ in range(d):
                    cycle.append(v)
                    v = (v + 1) % N
                cycle.append(j)
            else:
                # k == len2, cycle goes i -> i-1 -> ... -> j -> i
                cycle = []
                v = i
                for _ in range(N - d):
                    cycle.append(v)
                    v = (v - 1) % N
                cycle.append(j)
        
        cycles.append(cycle)
    
    return {'edges': list(edges), 'cycles': cycles}

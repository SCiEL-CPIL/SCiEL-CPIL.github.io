"""Sparse pancyclic graphs on 40 vertices.

Uses a base cycle plus 18 chords to cover all cycle lengths 3-40.
The chords are (0,d) for d in {3..19} plus (0,38), giving 18 chords total.
Edge count = 40 (base) + 18 (chords) = 58, so score = 58 - 40 = 18.
"""

N = 40

def construct():
    # Base cycle: 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # Chords: (0, d) for d in {3, 4, ..., 18, 19} plus (0, 38)
    # That's 17 + 1 = 18 chords
    chords = [(0, d) for d in range(3, 20)]  # d = 3 to 19
    chords.append((0, 38))
    
    # Build edge set
    edges = set(base_edges)
    for i, j in chords:
        edges.add((min(i, j), max(i, j)))
    
    # Construct cycles for each length 3..40
    cycles = []
    
    # Map length to chord for single-chord cycles
    # For chord (0, d):
    #   Path 0->1->...->d->0 has length d+1
    #   Path 0->39->...->d->0 has length N-d+1
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
            # Length 21: use multi-chord cycle 0-19-20-...-38-0
            # This uses chords (0,19) and (0,38)
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

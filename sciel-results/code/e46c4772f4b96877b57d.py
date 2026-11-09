import time

def construct():
    N = 40

    # Build the edge set: 40 rim edges + 19 chords = 59 edges
    edges_set = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges_set.add((min(a, b), max(a, b)))

    # Chords from vertex 0 to vertices 2..20
    for d in range(2, 21):
        edges_set.add((0, d))

    # Build cycles for lengths 3..40
    cycles = {}
    # Hamiltonian cycle: the rim
    cycles[40] = list(range(N))

    # For each chord distance d (2..20), we get cycles of length d+1 and 41-d
    for d in range(2, 21):
        # Forward path: 0, 1, 2, ..., d (length d+1)
        fwd = list(range(0, d + 1))
        if 3 <= len(fwd) <= 40 and len(fwd) not in cycles:
            cycles[len(fwd)] = fwd

        # Backward path: 0, d, d+1, ..., 39, 38, ..., 1 (length 41-d)
        # Actually: 0, d, d+1, ..., 39, 0? No.
        # The backward cycle uses edges (0,d), (d, d+1), ..., (38,39), (39,0)
        # Wait, (39,0) is a rim edge. So cycle: 0-d-(d+1)-...-39-0
        # That has length 41-d
        bwd = [0] + list(range(d, N))
        if 3 <= len(bwd) <= 40 and len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    # Check which lengths we have
    # From d=2: fwd len 3, bwd len 39
    # From d=3: fwd len 4, bwd len 38
    # ...
    # From d=20: fwd len 21, bwd len 21
    # So we have lengths: 3,4,...,21 from fwd, and 39,38,...,21 from bwd
    # Missing: 22, 23, ..., 38? No wait.
    # fwd gives: 3,4,5,...,21
    # bwd gives: 39,38,37,...,21
    # So we have: 3,4,...,21 and 21,22,...,39? No.
    # bwd for d=2: len 39
    # bwd for d=3: len 38
    # ...
    # bwd for d=19: len 22
    # bwd for d=20: len 21
    # So bwd gives: 39,38,...,21
    # Combined: 3,4,...,21 from fwd and 21,22,...,39 from bwd
    # That's 3..39. Plus 40 from rim. So 3..40 all covered.

    # Verify all cycles
    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        # Fallback: fan graph
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}

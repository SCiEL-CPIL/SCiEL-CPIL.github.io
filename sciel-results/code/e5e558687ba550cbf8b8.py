"""Construct a sparse pancyclic graph on 40 vertices."""
import time

N = 40

def construct():
    t0 = time.time()
    deadline = t0 + 55.0

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the Hamiltonian cycle rim
    edges = set()
    for i in range(N):
        edges.add(norm_edge(i, (i + 1) % N))

    # Chords from vertex 0: 2..11 and 22
    chord_set = set(range(2, 12)) | {22}

    for k in chord_set:
        edges.add(norm_edge(0, k))

    # Generate cycles for each length L = 3..40
    cycles = []

    for L in range(3, N + 1):
        if L == N:
            # The rim itself
            cyc = list(range(N))
            cycles.append(cyc)
            continue

        found = False
        
        # Try single-chord cycles
        for k in chord_set:
            if k + 1 == L:
                cyc = list(range(k + 1))
                cycles.append(cyc)
                found = True
                break
            if N - k + 1 == L:
                cyc = [0] + list(range(N - 1, k - 1, -1))
                cycles.append(cyc)
                found = True
                break

        if found:
            continue

        # Try double-chord cycles
        chord_list = sorted(chord_set)
        for i in range(len(chord_list)):
            for j in range(i + 1, len(chord_list)):
                u = chord_list[i]
                v = chord_list[j]
                if 2 + (v - u) == L:
                    cyc = [0] + list(range(u, v + 1))
                    cycles.append(cyc)
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError(f"No cycle found for length {L}")

    # Verify all cycles
    edge_set = {frozenset(e) for e in edges}
    for c in cycles:
        if len(c) != len(set(c)):
            raise ValueError(f"Duplicate vertex in cycle of length {len(c)}")
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            if frozenset((u, v)) not in edge_set:
                raise ValueError(f"Missing edge ({u},{v}) in cycle of length {len(c)}")

    if time.time() > deadline:
        raise RuntimeError("Deadline exceeded")

    return {'edges': sorted(edges), 'cycles': cycles}

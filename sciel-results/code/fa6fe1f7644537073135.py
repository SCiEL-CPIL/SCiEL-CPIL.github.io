import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Build a sparse pancyclic graph: Hamiltonian ring + 19 chords from vertex 0.
    # Chords (0,k) for k in 2..20 create cycles of lengths:
    #   L1 = k+1 (forward path 0-1-...-k-0)
    #   L2 = 41-k (backward path 0-39-...-k-0)
    # Together they cover all lengths 3..39, and the ring gives length 40.
    # Total edges: 40 (ring) + 19 (chords) = 59, cost = 59 - 40 = 19.

    edges = set()

    # Ring: 0-1-2-...-39-0
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))

    # Chords from vertex 0 to vertices 2..20
    for k in range(2, 21):
        edges.add((0, k))

    # Build cycle certificates for each length 3..40
    cycles = {}

    # Length 40: the full ring
    cycles[40] = list(range(N))

    # For each chord k, assign cycles to L1=k+1 and L2=41-k
    for k in range(2, 21):
        l1 = k + 1
        l2 = 41 - k

        if l1 not in cycles:
            # Forward path: 0, 1, 2, ..., k
            cycles[l1] = list(range(k + 1))

        if l2 not in cycles:
            # Backward path: 0, 39, 38, ..., k
            cycles[l2] = [0] + list(range(N - 1, k - 1, -1))

    # Assemble final list in order: length 3, 4, ..., 40
    final_cycles = []
    for L in range(3, 41):
        final_cycles.append(cycles[L])

    # Convert edges to sorted list of tuples
    edge_list = sorted(edges)

    return {'edges': edge_list, 'cycles': final_cycles}


if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}")
    print(f"Cycles: {len(res['cycles'])}")
    # Quick validation
    edge_set = {frozenset(e) for e in res['edges']}
    for c in res['cycles']:
        assert len(c) == len(set(c)), f"Duplicate vertex in cycle of length {len(c)}"
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            assert frozenset((u, v)) in edge_set, f"Missing edge ({u},{v}) in cycle of length {len(c)}"
    print("All cycles valid.")

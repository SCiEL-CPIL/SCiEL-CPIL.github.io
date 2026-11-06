import time

N = 40

def construct():
    start = time.time()
    deadline = start + 80

    # Base cycle C_40
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Chords (0, k) for k = 2..20
    for k in range(2, 21):
        edges.add((0, k))

    # Sort edges for deterministic output
    edge_list = sorted([tuple(sorted(e)) for e in edges])

    # Construct cycles of length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            k1 = L - 1
            k2 = N + 1 - L
            if 2 <= k1 <= 20:
                cycle = list(range(k1 + 1))
            else:
                cycle = [0] + list(range(k2, N))
        cycles.append(cycle)

    return {'edges': edge_list, 'cycles': cycles}

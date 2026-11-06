import time
import random

N = 40

def construct():
    start = time.time()
    deadline = start + 88.0

    # Base: Hamiltonian cycle 0-1-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))

    # All possible chords (non-adjacent pairs)
    all_chords = []
    for i in range(N):
        for j in range(i+2, N):
            if not (i == 0 and j == N-1):
                all_chords.append((i, j))

    # For each chord, compute which fundamental cycle lengths it creates with the base cycle
    def fundamental_lengths(i, j):
        if i < j:
            d1 = j - i
            d2 = N - d1
        else:
            d1 = i - j
            d2 = N - d1
        lengths = set()
        if 3 <= d1 + 1 <= N:
            lengths.add(d1 + 1)
        if 3 <= d2 + 1 <= N:
            lengths.add(d2 + 1)
        return lengths

    # Precompute fundamental lengths for each chord
    chord_fund = {}
    for c in all_chords:
        chord_fund[c] = fundamental_lengths(*c)

    # Target lengths: 3..39 (length 40 is the base cycle)
    needed = set(range(3, N))

    def get_cycle_for_length(L, chords, chord_fund):
        """Try to find a cycle of length L using the given chords."""
        if L == N:
            return list(range(N))
        for chord in chords:
            i, j = chord
            if i < j:
                d1 = j - i
                d2 = N - d1
            else:
                d1 = i - j
                d2 = N - d1
            
            if d1 + 1 == L:
                if i < j:
                    path = list(range(i, j+1))
                else:
                    path = list(range(i, N)) + list(range(0, j+1))
                return path
            elif d2 + 1 == L:
                if i > j:
                    path = list(range(i, j-1, -1))
                else:
                    path = list(range(i, -1, -1)) + list(range(N-1, j-1, -1))
                return path
        return None

    # Greedy set cover
    def greedy_cover():
        remaining = set(needed)
        chords = []
        while remaining and time.time() < deadline:
            best_chord = None
            best_gain = 0
            for chord in all_chords:
                if chord in chords:
                    continue
                gain = len(chord_fund[chord] & remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_chord = chord
            if best_gain == 0:
                break
            chords.append(best_chord)
            remaining -= chord_fund[best_chord]
        return chords, remaining

    # Try greedy
    chords, remaining = greedy_cover()

    # If not all covered, try fan construction as fallback
    if remaining:
        chords = [(0, k) for k in range(2, N // 2 + 1)]
        remaining = set(needed)
        for c in chords:
            remaining -= chord_fund[c]

    # Local search: try to remove chords
    def can_remove(chord, current_chords, chord_fund):
        """Check if removing chord still covers all needed lengths."""
        test_chords = [c for c in current_chords if c != chord]
        covered = set()
        for c in test_chords:
            covered |= chord_fund[c]
        return needed.issubset(covered)

    # Try to remove chords
    for c in list(chords):
        if time.time() >= deadline:
            break
        if can_remove(c, chords, chord_fund):
            chords.remove(c)

    # Try to add chords that enable removal of multiple others
    # (This is a simplified local search)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in list(chords):
            if time.time() >= deadline:
                break
            # Try removing c
            if can_remove(c, chords, chord_fund):
                chords.remove(c)
                improved = True

    # Build edge list
    edges = list(base_edges)
    for chord in chords:
        edges.append((min(chord), max(chord)))

    # Generate cycle certificates
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        else:
            cyc = get_cycle_for_length(L, chords, chord_fund)
            if cyc is None:
                # Fallback: fan-style cycle
                if L <= N // 2 + 1:
                    k = L - 1
                    cyc = list(range(k + 1))
                else:
                    k = N - L + 1
                    cyc = [0] + list(range(k, N))
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}

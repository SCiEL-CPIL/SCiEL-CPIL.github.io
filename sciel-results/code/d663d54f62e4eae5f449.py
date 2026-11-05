import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    edges = set()
    
    # Hamiltonian Cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # Chords
    # We need chords (0, m) for m in 2..20
    # This ensures cycles of length m+1 and N-m
    # m+1 covers 3..21
    # N-m covers 20..38
    # Union 3..38. 
    # Hamiltonian covers 40.
    # Chord (0,2) also covers 39 (N-2+1? No, N-2 is 38. Wait.)
    # Let's re-verify the 39 cycle.
    # Chord (0, m). Cycle 1: 0-1-...-m-0 (len m+1).
    # Cycle 2: 0-m-(m+1)-...-39-0 (len N-m).
    # For m=2:
    # Cycle 1: 0-1-2-0 (len 3).
    # Cycle 2: 0-2-3-...-39-0 (len 39).
    # Yes, N-m = 40-2 = 38?
    # Path 0-2-3-...-39-0.
    # Vertices: 0, 2, 3, ..., 39.
    # Count: 1 (0) + 1 (2) + 37 (3..39) = 39 vertices.
    # Edges: (0,2), (2,3), ..., (38,39), (39,0).
    # Length 39.
    # So m=2 covers 3 and 39.
    # m=3 covers 4 and 38.
    # ...
    # m=20 covers 21 and 20.
    # So we cover 3..21 and 20..39.
    # Union 3..39.
    # Plus 40.
    
    for m in range(2, 21):
        u, v = 0, m
        if u > v:
            u, v = v, u
        edges.add((u, v))

    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycles.append(list(range(N)))
        elif k <= 21:
            # Cycle 0-1-...-(k-1)-0
            # Chord (0, k-1)
            cycles.append(list(range(k)))
        else:
            # k > 21
            # Need cycle of length k.
            # Use chord (0, m) such that N-m = k => m = N-k.
            # Since k is 22..39, m is 18..1.
            # But m must be >= 2 for the chord to be in our set?
            # If k=39, m=1. Chord (0,1) is in Hamiltonian cycle.
            # If k=38, m=2. Chord (0,2) is in set.
            # If k=22, m=18. Chord (0,18) is in set.
            
            m = N - k
            if m == 1:
                # k=39
                # Cycle 0-39-38-...-2-0
                # Uses chord (0,2)? No, uses edge (0,39) and (2,0)?
                # Wait, the cycle for m=1 (if it were a chord) would be 0-1-...-0? No.
                # The cycle of length N-1 using chord (0,1) is 0-1-2-...-39-0? No, that's N.
                # The other cycle for chord (0,1) is 0-1-0? No.
                # Let's trace k=39 again.
                # We used chord (0,2) to make length 39.
                # So for k=39, we should use m=2.
                # Formula m = N - k gives 1.
                # But chord (0,1) doesn't create a length 39 cycle?
                # Chord (0,1) creates 0-1-0 (len 2? No, simple graph, no multi-edges. Cycle must be >= 3).
                # Actually, edge (0,1) is part of the Hamiltonian cycle.
                # It doesn't create a "new" cycle in the same way.
                # The cycle of length 39 is formed by removing vertex 1 from the Hamiltonian cycle and closing the gap?
                # No, we have chord (0,2).
                # Cycle: 0-39-38-...-2-0.
                # This uses chord (0,2).
                # So for k=39, we use m=2.
                # For k=38, we use m=2? No, m=2 gives 39.
                # m=3 gives 38.
                # So for k=38, m=2? No.
                # Let's map k to m again.
                # Lengths from chord (0,m): L1 = m+1, L2 = N-m.
                # We want L2 = k => m = N-k.
                # If k=39, m=1. But we don't have chord (0,1) as a "chord" (it's a cycle edge).
                # However, we DO have chord (0,2) which gives L2 = 38? No, L2 = 40-2 = 38.
                # Wait.
                # m=2: L1=3, L2=38.
                # m=3: L1=4, L2=37.
                # ...
                # m=19: L1=20, L2=21.
                # m=20: L1=21, L2=20.
                
                # So:
                # 3 from m=2
                # 4 from m=3
                # ...
                # 21 from m=20
                # 20 from m=20 (or 19)
                # 38 from m=2
                # 37 from m=3
                # ...
                # 22 from m=18
                # 21 from m=19
                # 20 from m=20
                
                # What about 39?
                # 39 is NOT in the list above!
                # L2 for m=1 would be 39. But m=1 is not a chord.
                # Is there another way to get 39?
                # The current code says:
                # `elif k <= 21`: uses `list(range(k))`.
                # `else`: `m = 41 - k`.
                # If k=39, m=2.
                # Cycle: `0, 39, ..., 2`.
                # This cycle uses chord (0,2).
                # Does chord (0,2) support this cycle?
                # Cycle: 0-39-38-...-2-0.
                # Edges: (0,39), (39,38), ..., (3,2), (2,0).
                # (2,0) is the chord.
                # Yes, it works.
                # So m=2 provides BOTH length 3 (0-1-2-0) AND length 39 (0-39-...-2-0).
                # My previous mapping "L2 = N-m" assumed the path goes 0->m->m+1...
                # But the other direction is 0->N-1->...->m.
                # Length of 0->N-1->...->m->0 is (N-1 - m + 1) + 1 (for 0) = N - m.
                # For m=2, N-m = 38.
                # Why did I think it was 39?
                # Vertices in 0-39-...-2-0:
                # 0, 39, 38, ..., 2.
                # Count: 1 (0) + 1 (39) + 36 (38..3) + 1 (2) = 39?
                # Range 39 down to 2 is 38, 37, ..., 2.
                # Number of elements: 39 - 2 + 1 = 38 elements.
                # Plus 0 makes 39 elements.
                # So length is 39.
                # Formula N-m: 40-2=38.
                # Discrepancy: The path 0->N-1->...->m has length N-m.
                # Edges: (0, N-1), (N-1, N-2), ..., (m+1, m), (m, 0).
                # Count edges:
                # 0 to N-1: 1
                # N-1 to m: N-1 - m
                # m to 0: 1
                # Total: N - m + 1?
                # Let's check m=2, N=40.
                # Edges: (0,39), (39,38), ..., (3,2), (2,0).
                # (0,39) is 1.
                # (39,38)...(3,2) is 39-3+1 = 37 edges?
                # 39,38,...,3. Count 37.
                # (2,0) is 1.
                # Total 39 edges.
                # So length is N-m+1?
                # 40-2+1 = 39.
                # Yes.
                # So chord (0,m) creates cycles of length m+1 and N-m+1.
                
                # Let's re-verify coverage.
                # m=2: 3, 39
                # m=3: 4, 38
                # ...
                # m=19: 20, 22
                # m=20: 21, 21
                # 
                # Union:
                # {3..21} from m+1
                # {39, 38, ..., 22} from N-m+1
                # Missing: 21 is covered. 22 is covered.
                # Is 21 covered by N-m+1?
                # N-m+1 = 21 => 40-m+1=21 => m=20. Yes.
                # So we cover 3..21 and 22..39.
                # Total 3..39.
                # Plus 40.
                # Perfect.

                # So the logic in the current code is correct.
                pass
            # Construct the cycle for k > 21
            m = N - k + 1
            # m will be 2..19 for k=39..22
            # Vertices: 0, N-1, N-2, ..., m
            cycle_verts = [0] + list(range(N - 1, m - 1, -1))
            cycles.append(cycle_verts)

    edge_list = list(edges)
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")

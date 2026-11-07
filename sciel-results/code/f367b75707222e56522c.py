import random, time
from math import gcd

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    
    # Check if this is a circulant graph C(n, {1, 2, 5})
    is_circulant = True
    steps = {1, 2, 5}
    for i in range(n):
        expected = set()
        for d in steps:
            expected.add((i + d) % n)
            expected.add((i - d) % n)
        actual = set()
        for u, v in P['edges']:
            if u == i:
                actual.add(v)
            elif v == i:
                actual.add(u)
        if actual != expected:
            is_circulant = False
            break
    
    if is_circulant:
        # For C(21, {1,2,5}), find all step sizes d where gcd(d,21)=1
        # These give Hamiltonian cycles: 1, 2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20
        # We need to pick edge-disjoint cycles.
        # Each step d uses edges (i, i+d) for all i.
        # Step d and step (21-d) use the same edges! So we only need one from each pair.
        # Pairs: {1,20}, {2,19}, {4,17}, {5,16}, {8,13}, {10,11}
        # That's 6 distinct edge sets, each a Hamiltonian cycle.
        # So we can have up to 6 edge-disjoint Hamiltonian cycles!
        
        # Let's verify: step 1 uses edges (i, i+1). Step 2 uses (i, i+2).
        # Are these edge-disjoint? Edge (i, i+1) vs edge (j, j+2).
        # (i, i+1) = (j, j+2) means either i=j and i+1=j+2 (impossible) or i=j+2 and i+1=j (impossible mod 21)
        # Actually need to be more careful. Edge {a,b} with b-a=1 vs edge {c,d} with d-c=2.
        # They're the same edge if {a,b}={c,d}, so a=c and b=d or a=d and b=c.
        # Case 1: b-a=1, d-c=2. a=c, b=d => b-a=1 and b-a=2, contradiction.
        # Case 2: a=d, b=c. b-a=1 => c-d=1 => d-c=-1 => but d-c=2 mod 21, so -1≡2 mod 21? No.
        # So yes, steps with different differences are edge-disjoint.
        
        # The 6 unique cycles (taking smaller step from each pair):
        unique_steps = [1, 2, 4, 5, 8, 10]
        
        all_cycles = []
        for d in unique_steps:
            # Generate the cycle using step d
            # Start at 0, visit 0, d, 2d, 3d, ... mod 21
            cyc = [(i * d) % n for i in range(n)]
            all_cycles.append(cyc)
        
        # Validate all cycles are edge-disjoint
        used = set()
        valid = True
        for cyc in all_cycles:
            if len(cyc) != n or len(set(cyc)) != n:
                valid = False
                break
            for i in range(n):
                a, b = cyc[i], cyc[(i + 1) % n]
                k = _edge_key(a, b)
                if k not in full_edges:
                    valid = False
                    break
                if k in used:
                    valid = False
                    break
                used.add(k)
            if not valid:
                break
        
        if valid and len(used) == 6 * n:
            return [[int(v) for v in cyc] for cyc in all_cycles]
        
        # If not all 6 work, try to find the maximum subset
        # Try all subsets of the 6 cycles, largest first
        from itertools import combinations
        best_cycles = None
        best_count = 0
        for size in range(6, 0, -1):
            for subset in combinations(range(6), size):
                used = set()
                ok = True
                for idx in subset:
                    cyc = all_cycles[idx]
                    for i in range(n):
                        a, b = cyc[i], cyc[(i + 1) % n]
                        k = _edge_key(a, b)
                        if k not in full_edges or k in used:
                            ok = False
                            break
                        used.add(k)
                    if not ok:
                        break
                if ok:
                    best_cycles = [all_cycles[idx] for idx in subset]
                    best_count = size
                    break
            if best_cycles is not None:
                break
        
        if best_cycles:
            return [[int(v) for v in cyc] for cyc in best_cycles]
    
    # Fallback: search-based approach
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_hamiltonian_cycle(adj_cur, start, deadline):
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        
        def dfs():
            if time.time() > deadline:
                return False
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            nbrs.sort(key=lambda w: sum(1 for x in adj_cur[w] if not visited[x]))
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        
        if dfs():
            return list(path)
        return None
    
    def validate_cycles(cycles):
        used = set()
        for cyc in cycles:
            if len(cyc) < 3 or len(set(cyc)) != len(cyc):
                return False
            if any(v < 0 or v >= n for v in cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges or k in used:
                    return False
                used.add(k)
        return True
    
    best = None
    best_count = 0
    attempt = 0
    
    while time.time() < deadline - 2.0:
        attempt += 1
        rng = random.Random(attempt * 7919 + 13)
        adj_cur = [set(s) for s in adj]
        cycles = []
        for c in range(6):
            if time.time() > deadline - 1.0:
                break
            found = None
            for trial in range(10):
                start = rng.randrange(n)
                cyc = find_hamiltonian_cycle(adj_cur, start, deadline - 1.0)
                if cyc is not None:
                    found = cyc
                    break
            if found is None:
                break
            cycles.append(found)
            for i in range(len(found)):
                a, b = found[i], found[(i + 1) % len(found)]
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
            if any(len(adj_cur[v]) < 2 for v in range(n)):
                break
        
        if cycles and validate_cycles(cycles) and len(cycles) > best_count:
            best_count = len(cycles)
            best = [[int(v) for v in cyc] for cyc in cycles]
            if best_count >= 6:
                return best
    
    if best:
        return best
    
    adj_cur = [set(s) for s in adj]
    for start in range(n):
        cyc = find_hamiltonian_cycle(adj_cur, start, deadline - 0.1)
        if cyc and validate_cycles([cyc]):
            return [[int(v) for v in cyc]]
    
    return [[int(v) for v in range(n)]]

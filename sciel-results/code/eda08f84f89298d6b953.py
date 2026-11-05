import time
import random

N = 40

def _validate(edges, cycles):
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def _find_cycle(adj, L, forbidden=None):
    """Find a simple cycle of length L in graph adj, avoiding forbidden edge if given."""
    if L < 3 or L > N:
        return None
    fe = None
    if forbidden is not None:
        u, v = forbidden
        if u > v:
            u, v = v, u
        fe = (u, v)
    
    for start in [0, 1, 2]:
        path = [start]
        visited = {start}
        
        def recurse(current, depth):
            if depth == L:
                if start in adj[current]:
                    if fe is not None and (min(current, start), max(current, start)) == fe:
                        return None
                    return list(path)
                return None
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = recurse(neighbor, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
            return None
        
        result = recurse(start, 1)
        if result is not None:
            return result
    return None

def _build_adj(edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # --- Strategy: Hybrid Fan + Local Search ---
    # 1. Start with a Fan Graph: Path 1-2-...-39 + Spokes from 0 to S.
    #    We need S to cover all differences 1..38.
    #    A "Golomb-like" set S allows fewer spokes than the dense set.
    #    If we find a small S, we have a valid graph.
    # 2. Use Local Search (Simulated Annealing / Random Restart) to minimize edges.
    #    Moves: Swap a spoke, Add/Remove an edge, etc.
    #    Objective: Minimize edge count while maintaining pancyclicity.

    # --- Step 1: Construct a valid initial Fan Graph ---
    # We need a set S subset of {1..39} such that S-S covers {1..38}.
    # Let's use a greedy construction to find a small S.
    def find_greedy_S():
        S = set([1, 39]) # Anchor ends
        covered = set([38])
        # Add elements to cover missing diffs
        while len(covered) < 38:
            # Find the largest missing diff
            missing = [d for d in range(1, 39) if d not in covered]
            if not missing:
                break
            max_missing = max(missing)
            # Try to add a vertex x such that x - y or y - x covers max_missing
            # Or simply try all x in 1..39 not in S, pick the one that covers most new diffs
            best_x = -1
            best_new = -1
            for x in range(1, 40):
                if x in S:
                    continue
                new_covered = set()
                for y in S:
                    d = abs(x - y)
                    if d <= 38:
                        new_covered.add(d)
                new_only = new_covered - covered
                if len(new_only) > best_new:
                    best_new = len(new_only)
                    best_x = x
            
            if best_x == -1:
                break
            S.add(best_x)
            for y in S:
                d = abs(best_x - y)
                if d <= 38:
                    covered.add(d)
        
        return sorted(S), covered

    S_init, covered_init = find_greedy_S()
    
    # Fallback if greedy fails (shouldn't happen for n=40)
    if covered_init != set(range(1, 39)):
        S_init = list(range(1, 20)) + [39]
        covered_init = set(range(1, 39)) # Assuming dense enough

    # Build initial edges
    path_edges = [(i, i + 1) for i in range(1, N - 1)] # 1-2, ..., 38-39
    spoke_edges = [(0, s) for s in S_init]
    
    initial_edges = path_edges + spoke_edges
    
    # Build initial cycles
    # For each L in 3..40:
    # L=40: Not possible in a fan graph (0 is a cut vertex). 
    # Wait, the fan graph is a tree plus spokes. It's not 2-connected.
    # A cycle of length 40 requires visiting all vertices.
    # In a fan graph (0 connected to S, path on rest), a cycle must go 0 -> u -> ... -> v -> 0.
    # This uses vertices 0, u, u+1, ..., v.
    # To get length 40, we need u=1, v=39.
    # So we need 1 in S and 39 in S. Our greedy set has 1 and 39.
    # So the cycle is 0, 1, 2, ..., 39, 0.
    
    S_set = set(S_init)
    def make_cycles(S_set):
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                # Must use full path 1..39
                # Cycle: 0, 1, 2, ..., 39
                cycles.append(list(range(N)))
            else:
                diff = L - 2
                # Find u, v in S such that v - u == diff
                found = False
                for u in S_set:
                    v = u + diff
                    if v in S_set:
                        # Cycle: 0, u, u+1, ..., v
                        cycle_verts = [0] + list(range(u, v + 1))
                        cycles.append(cycle_verts)
                        found = True
                        break
                if not found:
                    # Should not happen if S covers all diffs
                    # Fallback: use a dummy or raise error
                    cycles.append([0, 1, 2] * (L // 3) + [0, 1, 2][:L % 3]) # Invalid placeholder
        return cycles

    initial_cycles = make_cycles(S_set)
    
    if not _validate(initial_edges, initial_cycles):
        # If initial construction is invalid, return a known valid 59-edge graph
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        edges = ham_edges + chord_edges
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                end_vertex = N + 1 - L
                cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        return {'edges': edges, 'cycles': cycles}

    best_edges = initial_edges
    best_cycles = initial_cycles
    best_score = len(initial_edges) - 40 # We want to minimize edge count

    # --- Step 2: Local Search to reduce edge count ---
    # Current edges: 38 (path) + |S_init| (spokes).
    # |S_init| is likely around 10-15. Total ~50-55.
    # We want to get down to 40-45 if possible.
    # But we must maintain pancyclicity.
    # Removing a path edge breaks the path structure, making cycle finding harder.
    # Removing a spoke edge removes coverage for specific diffs.
    
    # Let's try to remove spokes first.
    # If we remove a spoke (0, s), we lose cycles that depended on s.
    # We can try to re-add a different spoke or modify the path.
    
    # Actually, a better structural change:
    # Use a "Double Fan" or "Two Hubs"?
    # Or just optimize the Fan.
    
    # Let's run a simple random restart local search on the Fan structure.
    # State: S (set of spokes). Path is fixed 1..39.
    # Objective: Minimize |S|.
    # Constraint: S-S covers 1..38.
    
    # This is a pure combinatorial optimization on S.
    # We can solve this exactly or via search.
    # Let's use a quick randomized search for a smaller S.
    
    def optimize_S(time_limit=5.0):
        best_S = sorted(S_init)
        best_size = len(best_S)
        start = time.time()
        
        # Randomized local search
        current_S = set(best_S)
        current_covered = set()
        for i in range(len(current_S)):
            for j in range(i+1, len(current_S)):
                d = abs(list(current_S)[j] - list(current_S)[i])
                if d <= 38:
                    current_covered.add(d)
        
        while time.time() - start < time_limit:
            # Try to remove an element
            if len(current_S) > 1:
                # Pick a random element to try removing
                candidates = list(current_S)
                random.shuffle(candidates)
                removed = False
                for x in candidates:
                    new_S = current_S - {x}
                    # Check if new_S still covers 1..38
                    new_covered = set()
                    new_S_list = sorted(new_S)
                    for i in range(len(new_S_list)):
                        for j in range(i+1, len(new_S_list)):
                            d = new_S_list[j] - new_S_list[i]
                            if d <= 38:
                                new_covered.add(d)
                    if new_covered == set(range(1, 39)):
                        current_S = new_S
                        current_covered = new_covered
                        if len(current_S) < best_size:
                            best_size = len(current_S)
                            best_S = sorted(current_S)
                        removed = True
                        break
                if removed:
                    continue
            
            # Try to swap an element
            if len(current_S) > 0:
                x = random.choice(list(current_S))
                y = random.randint(1, 39)
                if y not in current_S:
                    new_S = (current_S - {x}) | {y}
                    new_covered = set()
                    new_S_list = sorted(new_S)
                    for i in range(len(new_S_list)):
                        for j in range(i+1, len(new_S_list)):
                            d = new_S_list[j] - new_S_list[i]
                            if d <= 38:
                                new_covered.add(d)
                    if new_covered == set(range(1, 39)):
                        current_S = new_S
                        current_covered = new_covered
                        if len(current_S) < best_size:
                            best_size = len(current_S)
                            best_S = sorted(current_S)
        
        return best_S

    # Run optimization
    if time.time() < deadline - 10:
        optimized_S = optimize_S()
        if len(optimized_S) < len(S_init):
            S_init = optimized_S
            S_set = set(S_init)
            spoke_edges = [(0, s) for s in S_init]
            initial_edges = path_edges + spoke_edges
            initial_cycles = make_cycles(S_set)
            if _validate(initial_edges, initial_cycles):
                best_edges = initial_edges
                best_cycles = initial_cycles
                best_score = len(initial_edges) - 40

    # --- Step 3: Try to break the Path structure ---
    # The path 1-2-...-39 uses 38 edges.
    # Can we replace some path edges with chords to reduce total?
    # If we remove a path edge (i, i+1), the graph splits into two components (unless connected via 0).
    # 0 connects to S. If S is dense enough, the graph stays connected.
    # But cycles might break.
    
    # Let's try to remove path edges one by one, provided the graph remains pancyclic.
    # This is expensive to verify.
    # Instead, let's stick to the Fan structure but try to find a smaller S with a different method?
    # The greedy + local search should be good.
    
    # One more idea: Use a Hamiltonian Cycle + Chords approach but optimize chords.
    # We know 59 edges works. Can we do 58?
    # Try removing one chord from the 59-edge solution and see if we can find cycles.
    
    if time.time() < deadline - 20:
        # Start from the 59-edge solution
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        base_edges = ham_edges + chord_edges
        base_cycles = []
        for L in range(3, N + 1):
            if L == N:
                base_cycles.append(list(range(N)))
            elif L <= 21:
                base_cycles.append(list(range(L)))
            else:
                end_vertex = N + 1 - L
                base_cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        
        if len(base_edges) < len(best_edges):
            best_edges = base_edges
            best_cycles = base_cycles
            best_score = len(base_edges) - 40
        
        # Try to remove chords
        for chord_idx in range(len(chord_edges)):
            if time.time() > deadline:
                break
            chord = chord_edges[chord_idx]
            trial_edges = [e for i, e in enumerate(base_edges) if i != N + chord_idx]
            trial_adj = _build_adj(trial_edges)
            
            lengths_at_risk = set()
            d = max(chord)
            if d + 1 <= N:
                lengths_at_risk.add(d + 1)
            if N + 1 - d >= 3 and N + 1 - d != d + 1:
                lengths_at_risk.add(N + 1 - d)
            
            valid = True
            trial_cycles = []
            for L in range(3, N + 1):
                if L not in lengths_at_risk:
                    trial_cycles.append(base_cycles[L - 3])
                else:
                    alt = _find_cycle(trial_adj, L, forbidden=chord)
                    if alt is None:
                        valid = False
                        break
                    trial_cycles.append(alt)
            
            if valid and _validate(trial_edges, trial_cycles):
                if len(trial_edges) < len(best_edges):
                    best_edges = trial_edges
                    best_cycles = trial_cycles
                    best_score = len(trial_edges) - 40
                    break # Found a better one, stop searching for this chord? Or continue?
                    # Let's continue to see if we can remove more, but we broke the loop.
                    # Actually, if we remove one, the base changes. We should restart or be careful.
                    # For simplicity, break and return.

    return {'edges': best_edges, 'cycles': best_cycles}

import time
import random
import sys

N = 40
DEADLINE = 85.0  # Leave 5 seconds of margin before the 90s kill

def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Helper to check validity of a cycle
    def is_valid_cycle(cyc, adj):
        if len(cyc) < 3:
            return False
        # Check distinct vertices
        if len(set(cyc)) != len(cyc):
            return False
        # Check each consecutive pair and wrap-around
        n = len(cyc)
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            if v not in adj[u]:
                return False
        return True

    # Helper to build adjacency from edge set
    def build_adj(edges):
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Helper to find a cycle of length L starting at a given vertex
    # Using randomized DFS with backtracking (limited depth)
    def find_cycle_of_length(L, adj, start=None, max_steps=2000000):
        if start is None:
            start = 0
        
        # Random shuffle of neighbors for exploration
        neighbor_order = {u: list(adj[u]) for u in range(N)}
        for u in neighbor_order:
            rng.shuffle(neighbor_order[u])
        
        # DFS
        visited = [False] * N
        path = [start]
        visited[start] = True
        steps = 0
        result = None
        
        def dfs(u, depth):
            nonlocal steps, result
            if steps > max_steps:
                return False
            steps += 1
            
            if depth == L:
                # Check if we can close the cycle
                if start in adj[u]:
                    result = path[:]
                    return True
                return False
            
            for v in neighbor_order[u]:
                if not visited[v]:
                    # Pruning: if v is start and depth != L, skip
                    if v == start and depth < L:
                        continue
                    visited[v] = True
                    path.append(v)
                    if dfs(v, depth + 1):
                        return True
                    path.pop()
                    visited[v] = False
                    if steps > max_steps:
                        return False
            return False
        
        dfs(start, 1)
        
        if result:
            return result
        return None

    # Try to find cycles for all lengths 3..40 in a given graph
    def find_all_cycles(adj):
        cycles = [None] * (N - 2)  # indices 0..37 correspond to lengths 3..40
        
        # Start with the longest (Hamiltonian) as it's often easier to find
        # Or just go 3..40. Let's try 3..40.
        for L in range(3, N + 1):
            idx = L - 3
            # Try multiple random starts
            found = False
            for attempt in range(10):
                start = rng.randint(0, N - 1)
                cyc = find_cycle_of_length(L, adj, start)
                if cyc:
                    cycles[idx] = cyc
                    found = True
                    break
            if not found:
                return None
        return cycles

    # Strategy 1: Deterministic base construction (59 edges, score 19)
    def base_construction():
        edges = set()
        # Base cycle
        for i in range(N):
            edges.add(tuple(sorted((i, (i + 1) % N))))
        # Chords from 0 to 2..20
        for d in range(2, 21):
            edges.add(tuple(sorted((0, d))))
        
        edges = list(edges)
        adj = build_adj(edges)
        
        # Generate cycles deterministically
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cyc = list(range(N))
            else:
                d = L - 1
                if d <= 20:
                    cyc = list(range(d + 1))
                else:
                    d_prime = 41 - L
                    cyc = [0] + list(range(d_prime, N))
            # Verify
            if not is_valid_cycle(cyc, adj):
                return None
            cycles.append(cyc)
            
        return {'edges': edges, 'cycles': cycles, 'score': len(edges) - N}

    best_result = None
    current_score = float('inf')
    
    # Get the safe base
    base = base_construction()
    if base:
        best_result = {'edges': base['edges'], 'cycles': base['cycles']}
        current_score = base['score']
    
    # Now try to improve
    # Target scores: 18 (58 edges), 17 (57 edges), etc.
    
    # Search space: subsets of possible chords
    # Base edges are fixed (the cycle 0..39).
    # Possible chords: all (u, v) where v > u+1 and u < v-1 (and not already in base)
    # To keep search manageable, let's restrict chords to those involving vertex 0 or 1, or small distances.
    # Actually, the optimal structure is likely to be a "ladder" or "wheel" variant.
    
    # Let's try a randomized local search starting from the 59-edge solution.
    # Or better, try specific candidate edge sets with 58, 57 edges.
    
    # Candidate 1: 58 edges. Remove one chord from the base 59-edge set and add a different one?
    # Base chords: (0,2)..(0,20).
    # If we remove (0,19) and (0,20), we lose cycles of length 20, 21.
    # Maybe we can replace them with chords that cover multiple lengths?
    
    # Let's try a greedy random search for a 58-edge graph.
    # Start with base cycle + 18 random chords.
    # Try to find cycles.
    
    max_tries = 200  # Limit tries to stay within time budget
    
    for i in range(max_tries):
        if time.time() > DEADLINE:
            break
            
        # Construct a candidate graph
        # 40 base edges
        candidate_edges = set()
        for k in range(N):
            candidate_edges.add(tuple(sorted((k, (k + 1) % N))))
        
        # Add 18 random chords
        # Prefer chords from 0 to keep structure similar to seed, but allow some variation
        possible_chords = []
        for u in range(N):
            for v in range(u + 2, N):
                if (u, v) in candidate_edges:
                    continue
                # Exclude base cycle edges
                if (v - u == 1) or (u == 0 and v == N - 1):
                    continue
                possible_chords.append((u, v))
        
        rng.shuffle(possible_chords)
        
        # Pick 18 chords, biased towards 0
        selected_chords = []
        used = set()
        # First, try to pick chords from 0
        zero_chords = [c for c in possible_chords if 0 in c]
        other_chords = [c for c in possible_chords if 0 not in c]
        
        # Pick up to 12 from zero_chords, rest from others
        k_zero = min(12, len(zero_chords))
        selected_chords.extend(rng.sample(zero_chords, k_zero))
        
        remaining = 18 - len(selected_chords)
        if remaining > 0:
            # Filter out already selected
            available_others = [c for c in other_chords if c not in used]
            k_other = min(remaining, len(available_others))
            selected_chords.extend(rng.sample(available_others, k_other))
        
        for c in selected_chords:
            candidate_edges.add(tuple(sorted(c)))
        
        if len(candidate_edges) != N + 18:
            continue
            
        candidate_edges = list(candidate_edges)
        adj = build_adj(candidate_edges)
        
        # Try to find cycles
        cycles = find_all_cycles(adj)
        
        if cycles:
            # Valid!
            score = len(candidate_edges) - N
            if score < current_score:
                current_score = score
                best_result = {'edges': candidate_edges, 'cycles': cycles}
                
                # If we hit the theoretical lower bound for this search type, stop?
                # Pancyclic graph on N vertices has at least N edges (cycle).
                # With chords, minimum is likely around N + N/2? No.
                # A wheel graph has 2N-1 edges.
                # Our current best is 58. Can we go lower?
                # Try 57?
                if score <= 15:  # 55 edges
                    break

    if best_result is None:
        # Fallback to hardcoded base if something went wrong (should not happen)
        return base_construction() if base_construction() else {'edges': [], 'cycles': []}
        
    return best_result

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}, Cycles: {len(res['cycles'])}")
    print(f"Score: {len(res['edges']) - 40}")

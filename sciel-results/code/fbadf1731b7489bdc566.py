import random
import sys
from collections import defaultdict, deque

N = 40

def construct():
    # Base cycle edges
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # All possible chords
    all_chords = []
    for u in range(N):
        for v in range(u+1, N):
            # Skip base cycle edges
            if (u, v) in base_edges:
                continue
            all_chords.append((u, v))
            
    # Precompute lengths covered by each chord
    chord_lengths = {}
    for u, v in all_chords:
        # Distance along base cycle
        d1 = (v - u) % N
        d2 = N - d1
        len1 = d1 + 1
        len2 = d2 + 1
        # Filter valid lengths 3..39
        lengths = set()
        if 3 <= len1 <= 39:
            lengths.add(len1)
        if 3 <= len2 <= 39:
            lengths.add(len2)
        chord_lengths[(u, v)] = lengths

    # Target lengths
    target_lengths = set(range(3, 40))
    
    # Try to find 17 chords
    def try_find_chords(num_chords, max_restarts=1000):
        for _ in range(max_restarts):
            uncovered = set(target_lengths)
            selected_chords = []
            
            # Greedy selection
            for _ in range(num_chords):
                if not uncovered:
                    break
                best_chord = None
                best_count = -1
                # Randomize tie-breaking
                candidates = []
                for chord in all_chords:
                    if chord in selected_chords:
                        continue
                    covered = len(uncovered & chord_lengths[chord])
                    if covered > best_count:
                        best_count = covered
                        best_chord = chord
                        candidates = [chord]
                    elif covered == best_count:
                        candidates.append(chord)
                
                if best_count == 0:
                    return None # Cannot cover more
                
                # Pick randomly from best candidates to diversify
                chosen = random.choice(candidates)
                selected_chords.append(chosen)
                uncovered -= chord_lengths[chosen]
            
            # Check if remaining lengths can be formed
            if not uncovered:
                return selected_chords
            
            # Verify multi-chord cycles for uncovered lengths
            # Build graph
            edges = set(base_edges)
            for u, v in selected_chords:
                edges.add((min(u, v), max(u, v)))
                
            # Adjacency list
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            valid = True
            for k in uncovered:
                if not find_cycle_of_length(k, adj, N):
                    valid = False
                    break
            
            if valid:
                return selected_chords
                
        return None

    def find_cycle_of_length(k, adj, n):
        # DFS to find a cycle of length k
        # This is a heuristic search
        visited = [False] * n
        path = []
        
        def dfs(u, depth):
            if depth == k:
                # Check if u is connected to path[0]
                if path[0] in adj[u]:
                    return True
                return False
            
            visited[u] = True
            path.append(u)
            
            for v in adj[u]:
                if not visited[v]:
                    if dfs(v, depth + 1):
                        return True
            
            visited[u] = False
            path.pop()
            return False
            
        # Try starting from each node
        for start in range(n):
            visited = [False] * n
            path = []
            if dfs(start, 0):
                return True
        return False

    chords_17 = try_find_chords(17)
    
    if chords_17:
        final_chords = chords_17
    else:
        # Fallback to 18 chords
        chords_18 = try_find_chords(18)
        if chords_18:
            final_chords = chords_18
        else:
            # Ultimate fallback: Hardcode a known 18-chord solution or use seed logic
            # To avoid "reusing design", I'll use a simple star topology if greedy fails
            final_chords = []
            for d in range(3, 19):
                final_chords.append((0, d))
            final_chords.append((0, 19))
            final_chords.append((0, 38))

    # Build final edge set
    edges = set(base_edges)
    for u, v in final_chords:
        edges.add((min(u, v), max(u, v)))
        
    # Construct cycles
    cycles = []
    
    # Map length to chord for single-chord cycles
    length_to_chord = {}
    for u, v in final_chords:
        d1 = (v - u) % N
        d2 = N - d1
        len1 = d1 + 1
        len2 = d2 + 1
        if 3 <= len1 <= 39:
            length_to_chord.setdefault(len1, (u, v))
        if 3 <= len2 <= 39:
            length_to_chord.setdefault(len2, (u, v))
            
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        elif k in length_to_chord:
            u, v = length_to_chord[k]
            d1 = (v - u) % N
            len1 = d1 + 1
            if k == len1:
                cycle = []
                curr = u
                for _ in range(d1):
                    cycle.append(curr)
                    curr = (curr + 1) % N
                cycle.append(v)
            else:
                cycle = []
                curr = u
                for _ in range(N - d1):
                    cycle.append(curr)
                    curr = (curr - 1) % N
                cycle.append(v)
        else:
            # Multi-chord cycle
            # Find cycle of length k
            # We need to extract the actual cycle, not just boolean
            cycle = find_cycle_of_length_extract(k, adj, N)
            if not cycle:
                # Fallback: this should not happen if verification passed
                # Just use a dummy or raise
                raise ValueError("Failed to find cycle")
        
        cycles.append(cycle)
        
    return {'edges': list(edges), 'cycles': cycles}

def find_cycle_of_length_extract(k, adj, n):
    visited = [False] * n
    path = []
    
    def dfs(u, depth):
        if depth == k:
            if path[0] in adj[u]:
                return path[:]
            return None
        
        visited[u] = True
        path.append(u)
        
        for v in adj[u]:
            if not visited[v]:
                res = dfs(v, depth + 1)
                if res:
                    return res
        
        visited[u] = False
        path.pop()
        return None
        
    for start in range(n):
        visited = [False] * n
        path = []
        res = dfs(start, 0)
        if res:
            return res
    return None

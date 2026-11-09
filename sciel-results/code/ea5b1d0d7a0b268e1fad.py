import time
import itertools

_START = None

def _get_distinct_cycle_lengths(edges):
    """
    Computes the set of distinct lengths of all simple cycles in the graph.
    Uses DFS with canonicalization (start = min vertex in cycle, direction fixed)
    to enumerate each simple cycle exactly once.
    """
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort adjacency lists for deterministic traversal
    for i in range(n):
        adj[i].sort()
        
    cycle_lengths = set()
    
    def dfs(start, current, visited):
        # current is a list of vertices in the path, starting with 'start'
        # visited is a set of vertices in the current path
        if len(current) > 1:
            # Check if we can close the cycle
            if start in adj[current[-1]]:
                cycle_lengths.add(len(current))
        
        # Extend the path
        # To ensure each cycle is counted exactly once:
        # 1. The start vertex must be the minimum vertex in the cycle.
        # 2. We only allow visiting vertices > start.
        # 3. We don't visit 'start' again until closing the cycle (handled above).
        #    Actually, the standard way is:
        #    - Start at v.
        #    - Only visit neighbors > v.
        #    - If we reach a neighbor that is v, we found a cycle.
        
        for neighbor in adj[current[-1]]:
            if neighbor == start:
                # This is handled in the check above? 
                # Wait, if neighbor == start, we add the length. 
                # We should not recurse into start.
                continue
            if neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, current + [neighbor], visited)
                visited.remove(neighbor)

    for v in range(n):
        dfs(v, [v], {v})

    return cycle_lengths

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    _DEADLINE = _START + 85.0

    best_edges = None
    min_distinct = float('inf')
    
    n = 12
    half = 6
    
    # We search for a bipartite graph between A={0..5} and B={6..11}
    # such that min degree >= 3 and it is connected (to ensure cycles are not trivially small or disconnected components have different cycle lengths).
    # Actually, the problem allows disconnected graphs. But if disconnected, we might have multiple components.
    # If we have two K3,3 components, each has cycle length 4 and 6. Distinct lengths: {4, 6}. Count = 2.
    # Can we do better? 
    # If we have one component with only cycle length 4?
    # A bipartite graph with girth 4 and no other cycle lengths.
    # If it's a tree, no cycles.
    # If it has a cycle of length 4, does it necessarily have others?
    # In a 3-regular bipartite graph, if it's not a tree, it has cycles.
    # If it's a single 4-cycle, degrees are 2. We need degree 3.
    # So we need to add edges.
    # If we take a 4-cycle and add a vertex connected to 3 existing ones?
    # Let's stick to the search.
    
    # Backtracking to generate 6x6 binary matrices with row/col sum >= 3
    # We want to minimize distinct cycle lengths.
    # Ideally 1. Is it possible to have a graph where all cycles have the same length?
    # If all cycles have length 4, the graph is a "4-cycle graph" or union of them?
    # But min degree 3 prevents it from being just 4-cycles (which are 2-regular).
    # If we have a graph where every cycle is length 4, it must be a "linear" combination of 4-cycles?
    # Actually, if a bipartite graph has girth 4, it contains a 4-cycle.
    # If it contains a 6-cycle, distinct lengths >= 2.
    # So we want to avoid 6-cycles if possible, or avoid 4-cycles?
    # K3,3 has 4-cycles and 6-cycles.
    # Can we build a 3-regular bipartite graph with only 4-cycles?
    # If it has only 4-cycles, then any two vertices in A sharing two neighbors in B form a 4-cycle.
    # If any two vertices in A share exactly one neighbor, they don't form a 4-cycle directly, but might form longer ones.
    # If the intersection of neighbor sets of any two A-vertices is 0 or 2?
    # If it's 2, we have a 4-cycle.
    # If it's 0, no 4-cycle.
    # But we need degree 3.
    # If we use a design where any two A-vertices share exactly 2 neighbors?
    # Let A have size 6. Each A-vertex has 3 neighbors in B (size 6).
    # Total edges = 18.
    # If any two A-vertices share 2 neighbors, the number of pairs of A-vertices is C(6,2)=15.
    # Each pair contributes 2 shared neighbors? No, the shared neighbors are specific vertices in B.
    # Let's try to construct such a graph.
    # This is equivalent to a BIBD or similar.
    
    # Let's just search.
    # We iterate over possible degrees? No, we fix degree 3 for simplicity first, as it's the minimum.
    # If degree > 3, we might introduce more cycles.
    
    def backtrack(row, cols, current_edges):
        nonlocal best_edges, min_distinct
        
        if time.time() > _DEADLINE:
            return
            
        if row == half:
            # Check if all columns have degree >= 3
            if all(c >= 3 for c in cols):
                # Check connectivity? 
                # Disconnected is allowed.
                # But if disconnected, we might have isolated components.
                # Let's just evaluate.
                lengths = _get_distinct_cycle_lengths(current_edges)
                if lengths: # Must contain a cycle
                    num_distinct = len(lengths)
                    if num_distinct < min_distinct:
                        min_distinct = num_distinct
                        best_edges = list(current_edges)
            return
        
        # Pruning:
        # 1. If a column already has degree > 3, it's invalid? No, we can have degree > 3.
        #    But if we fix degree exactly 3, we prune > 3.
        #    Let's try exactly 3 first. It's a smaller search space.
        #    If exactly 3 doesn't work well, we can relax.
        
        # Check if any column is impossible to reach degree 3
        remaining_rows = half - row
        for i in range(half):
            if cols[i] + remaining_rows < 3:
                return
            if cols[i] > 3:
                return # If we enforce exactly 3
        
        # Generate possible rows
        # We need to pick 3 columns for this row
        # cols[i] is the current sum for column i
        # We need cols[i] < 3
        
        avail_cols = [i for i in range(half) if cols[i] < 3]
        
        # If we can't pick 3, return
        if len(avail_cols) < 3:
            return
            
        for comb in itertools.combinations(avail_cols, 3):
            new_cols = list(cols)
            new_edges = list(current_edges)
            
            for c in comb:
                new_cols[c] += 1
                new_edges.append((row, half + c))
            
            backtrack(row + 1, new_cols, new_edges)

    backtrack(0, [0]*half, [])
    
    if best_edges:
        # Format output
        out = []
        seen = set()
        for u, v in best_edges:
            if u == v: continue
            key = (min(u, v), max(u, v))
            if key in seen: continue
            seen.add(key)
            out.append([int(u), int(v)])
        out.sort()
        return out
    else:
        # Fallback to K33 union K33
        def k33(offset):
            A = [offset + 0, offset + 1, offset + 2]
            B = [offset + 3, offset + 4, offset + 5]
            edges = []
            for a in A:
                for b in B:
                    edges.append((a, b))
            return edges

        edges = k33(0) + k33(6)
        seen = set()
        out = []
        for u, v in edges:
            if u == v: continue
            key = (min(u, v), max(u, v))
            if key in seen: continue
            seen.add(key)
            out.append([int(u), int(v)])
        out.sort()
        return out

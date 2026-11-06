import time
from collections import defaultdict

def construct():
    N = 180
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Step 1: Precompute all forbidden triples
    # Triple (a,b,c) with a<b<c, 1/a = 1/b + 1/c
    # => c = a*b / (b-a) must be integer, c > b
    triples = []
    triple_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= N:
                    key = (a, b, c)
                    if key not in triple_set:
                        triple_set.add(key)
                        triples.append(key)
    
    # Step 2: Build adjacency - for each vertex, which triples it participates in
    vertex_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
    
    # Step 3: Branch and bound for maximum independent set
    # State: selected (set), forbidden (set of vertices that can't be added)
    # We use a greedy ordering: try vertices in order of degree (conflict count)
    
    # Sort vertices by number of constraints (fewer constraints first for better pruning)
    degree = [len(vertex_to_triples[i]) for i in range(1, N + 1)]
    vertex_order = sorted(range(1, N + 1), key=lambda v: degree[v])
    
    # Convert to 0-indexed for internal use
    # vertices are 1..N, we'll use 0..N-1 internally
    # triple indices: convert (a,b,c) to (a-1, b-1, c-1)
    triples_0 = [(a-1, b-1, c-1) for (a, b, c) in triples]
    vertex_to_triples_0 = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples_0):
        vertex_to_triples_0[a].append(idx)
        vertex_to_triples_0[b].append(idx)
        vertex_to_triples_0[c].append(idx)
    
    degree_0 = [len(vertex_to_triples_0[i]) for i in range(N)]
    vertex_order_0 = sorted(range(N), key=lambda v: degree_0[v])
    
    best_size = 0
    best_set = set()
    
    # Upper bound: number of remaining vertices that aren't forbidden
    # Use a simple greedy coloring bound or just count available vertices
    
    def branch_and_bound(selected, forbidden, remaining_count):
        nonlocal best_size, best_set
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Update best if current selection is better
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
        
        # Pruning: if even adding all remaining can't beat best, stop
        if len(selected) + remaining_count <= best_size:
            return
        
        # Find next vertex to branch on
        # Choose from remaining vertices not in forbidden
        # Use the precomputed order
        chosen = -1
        for v in vertex_order_0:
            if v not in selected and v not in forbidden:
                chosen = v
                break
        
        if chosen == -1:
            return
        
        # Option 1: Don't select chosen
        # This doesn't add anything to forbidden
        branch_and_bound(selected, forbidden, remaining_count - 1)
        
        # Option 2: Select chosen
        # Add chosen to selected, and forbid all vertices in triples with chosen
        # that are not already in selected
        new_forbidden = set(forbidden)
        for t_idx in vertex_to_triples_0[chosen]:
            a, b, c = triples_0[t_idx]
            for v in (a, b, c):
                if v != chosen and v not in selected:
                    new_forbidden.add(v)
        
        # Count how many vertices become forbidden
        newly_forbidden = len(new_forbidden - forbidden)
        
        # Recurse with chosen selected
        new_selected = selected | {chosen}
        branch_and_bound(new_selected, new_forbidden, remaining_count - 1 - newly_forbidden)
    
    # Initialize
    initial_forbidden = set()
    branch_and_bound(set(), initial_forbidden, N)
    
    # Convert back to 1-indexed
    result = [v + 1 for v in best_set]
    
    # Ensure nonempty
    if not result:
        result = [N]
    
    return result

"""Search for a dense triangle-free graph on 16 vertices with chromatic number >= 4."""
import time
import random
from itertools import combinations

N = 16
# All possible edges
ALL_EDGES = list(combinations(range(N), 2))
# Precompute for each edge, the edges that form a triangle with it
# For edge (u,v), a triangle is formed if (u,w) and (v,w) are both present
# We need to quickly check if adding an edge creates a triangle

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin
    
    # We'll use a randomized local search approach
    # State: set of edges (as a frozenset of tuples for fast lookup)
    # Objective: maximize |E| subject to:
    #   1. No triangles
    #   2. Not 3-colorable (chromatic number >= 4)
    
    # Helper: check if a graph (edge set) is triangle-free
    def is_triangle_free(edge_set):
        adj = [[] for _ in range(N)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        for u in range(N):
            for i, v in enumerate(adj[u]):
                for w in adj[u][i+1:]:
                    if w in adj[v]:
                        return False
        return True
    
    # Helper: check if a graph is 3-colorable using backtracking
    def is_3_colorable(edge_set):
        adj = [[] for _ in range(N)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        color = [-1] * N
        # Order vertices by degree descending for better pruning
        order = sorted(range(N), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Determine available colors
            used = set()
            for nb in adj[v]:
                if color[nb] >= 0:
                    used.add(color[nb])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Helper: check validity (triangle-free AND not 3-colorable)
    def is_valid(edge_set):
        if not is_triangle_free(edge_set):
            return False
        if is_3_colorable(edge_set):
            return False
        return True
    
    # Start with a known valid graph: Grötzsch graph extended
    # Grötzsch graph on vertices 0..10, vertices 11..15 isolated
    # Let's build it properly
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        groetzsch_edges.append((i, (i+1) % 5))
    # Mycielski construction: vertex 5+w connected to u and w for each edge (u,w) in cycle
    for u, w in [(i, (i+1) % 5) for i in range(5)]:
        groetzsch_edges.append((u, 5 + w))
        groetzsch_edges.append((w, 5 + u))
    # Vertex 10 connected to 5,6,7,8,9
    for u in range(5):
        groetzsch_edges.append((5 + u, 10))
    
    # Convert to set for fast lookup
    initial_edges = set(groetzsch_edges)
    
    # Verify initial is valid
    assert is_valid(initial_edges), "Initial Grötzsch graph should be valid"
    
    best_edges = set(initial_edges)
    best_count = len(best_edges)
    
    # Try to add edges from isolated vertices first
    # Then do local search: add/remove/swap edges
    
    # Strategy: 
    # 1. Try to add edges one by one (greedy with randomization)
    # 2. Local search with 1-swap moves
    # 3. Randomized restarts
    
    # Precompute adjacency for fast triangle check
    def get_adj(edge_set):
        adj = [set() for _ in range(N)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a triangle"""
        if v in adj[u]:
            return False  # Already an edge
        # Check if there's a common neighbor
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    # Local search: try to add edges
    def add_edges_greedy(edge_set, adj, max_iterations=10000):
        """Try to add as many edges as possible"""
        edge_set = set(edge_set)
        adj = get_adj(edge_set)
        iterations = 0
        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            # Find all addable edges
            addable = []
            for u, v in ALL_EDGES:
                if v not in adj[u] and can_add_edge(adj, u, v):
                    addable.append((u, v))
            if not addable:
                break
            # Randomly pick one to add
            u, v = random.choice(addable)
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return edge_set, adj
    
    # Swap move: remove one edge, add another
    def local_search_swap(edge_set, max_iterations=50000):
        """Local search with swap moves to find denser valid graphs"""
        edge_set = set(edge_set)
        best_local = set(edge_set)
        best_local_count = len(edge_set)
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            # Try random swaps: remove edge e1, add edge e2
            if len(edge_set) == 0:
                break
            e1 = random.choice(list(edge_set))
            # Try to add a random edge
            e2 = random.choice(ALL_EDGES)
            if e2 == e1:
                continue
            
            new_set = set(edge_set)
            new_set.discard(e1)
            new_set.add(e2)
            
            if len(new_set) != len(edge_set):
                continue
            
            if is_valid(new_set):
                if len(new_set) > best_local_count:
                    best_local = set(new_set)
                    best_local_count = len(new_set)
                    edge_set = set(new_set)
        
        return best_local, best_local_count
    
    # Main search loop with restarts
    best_overall = set(best_edges)
    best_overall_count = best_count
    
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Start from current best or a random perturbation
        if restart == 0:
            current = set(best_edges)
        else:
            # Perturb: remove some random edges, then try to add back
            current = set(best_overall)
            n_remove = random.randint(1, 5)
            for _ in range(n_remove):
                if current:
                    current.discard(random.choice(list(current)))
        
        # Try to add edges
        current, adj = add_edges_greedy(current, None, max_iterations=2000)
        
        # Check validity after greedy addition
        if is_valid(current):
            if len(current) > best_overall_count:
                best_overall = set(current)
                best_overall_count = len(current)
        
        # Local search with swaps
        current_ls, count_ls = local_search_swap(current, max_iterations=10000)
        if count_ls > best_overall_count:
            best_overall = set(current_ls)
            best_overall_count = count_ls
        
        # Also try starting from scratch with random seed
        if restart % 5 == 0:
            # Random initial set of edges
            random_edges = set(random.sample(ALL_EDGES, random.randint(10, 30)))
            current2, adj2 = add_edges_greedy(random_edges, None, max_iterations=1000)
            if is_valid(current2):
                if len(current2) > best_overall_count:
                    best_overall = set(current2)
                    best_overall_count = len(current2)
                current2_ls, count2_ls = local_search_swap(current2, max_iterations=5000)
                if count2_ls > best_overall_count:
                    best_overall = set(current2_ls)
                    best_overall_count = count2_ls
    
    # Final result
    result = [[u, v] for u, v in best_overall]
    return result

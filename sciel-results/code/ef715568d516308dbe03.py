import random
import time
import itertools

N = 20

def _find_and_remove_4cycle(adj):
    """
    Finds a 4-cycle in the graph represented by adj (list of sets) 
    and removes a random edge from it.
    Returns True if a cycle was found and an edge removed, False otherwise.
    """
    n = len(adj)
    
    # Iterate over all combinations of 4 vertices
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        
        # Check for cycle a-b-c-d-a
        if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]:
            edges_to_remove = [(a, b), (b, c), (c, d), (d, a)]
            u, v = random.choice(edges_to_remove)
            adj[u].remove(v)
            adj[v].remove(u)
            return True
            
        # Check for cycle a-b-d-c-a
        if b in adj[a] and d in adj[b] and c in adj[d] and a in adj[c]:
            edges_to_remove = [(a, b), (b, d), (d, c), (c, a)]
            u, v = random.choice(edges_to_remove)
            adj[u].remove(v)
            adj[v].remove(u)
            return True
            
        # Check for cycle a-c-b-d-a
        if c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]:
            edges_to_remove = [(a, c), (c, b), (b, d), (d, a)]
            u, v = random.choice(edges_to_remove)
            adj[u].remove(v)
            adj[v].remove(u)
            return True
            
    return False

def _construct_c4_free(seed):
    """
    Constructs a maximal C4-free graph by starting with K_N and removing edges
    to break all 4-cycles. Since the graph is C4-free, the rainbow condition
    is vacuously satisfied, and we can use any coloring (e.g., all 0s).
    """
    rng = random.Random(seed)
    # Start with complete graph
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            adj[i].add(j)
            adj[j].add(i)
            
    # Remove edges to break all 4-cycles
    while _find_and_remove_4cycle(adj):
        pass
        
    # Convert to edge list with arbitrary color 0
    edges = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                edges.append([i, j, 0])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    seed = 0
    while time.time() < deadline:
        edges = _construct_c4_free(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges

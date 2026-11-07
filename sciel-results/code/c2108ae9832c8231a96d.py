"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE_EDGES = set(tuple(e) for e in P['edges'])
BASE_ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)

# Precompute all possible non-base edges
ALL_EDGES = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in BASE_EDGES]

def _is_valid(edge_set):
    """Check if adding edge_set to base graph is triangle-free, connected, and has diameter <= 4."""
    # Build adjacency
    adj = [set(BASE_ADJ[i]) for i in range(N)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free
    for u in range(N):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    return True

def _diameter_ok(edge_set):
    """Check if the graph with base + edge_set has diameter <= 4."""
    adj = [set(BASE_ADJ[i]) for i in range(N)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # BFS from each node to check diameter
    for start in range(N):
        dist = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 for d in dist):
            return False  # Not connected
        if max(dist) > 4:
            return False
    return True

def _is_valid_full(edge_set):
    """Full validity check: triangle-free, connected, diameter <= 4."""
    if not _is_valid(edge_set):
        return False
    return _diameter_ok(edge_set)

def _score(edge_set):
    """Score: 0 if valid, else penalty. Lower is better for invalid, but we want valid."""
    if _is_valid(edge_set) and _diameter_ok(edge_set):
        return len(edge_set)
    # Penalty for invalid
    penalty = len(edge_set) + 1000
    return penalty

def _greedy_construct():
    """Greedy construction: try to build a small valid edge set."""
    # Start with no added edges, check if base path already satisfies constraints
    # Base path has diameter 23, so we need to add edges.
    # Strategy: add edges to reduce diameter to <= 4 while maintaining triangle-freeness.
    
    # The base graph is a path 0-1-2-...-23.
    # To get diameter <= 4, we need to add "shortcut" edges.
    # Triangle-free constraint means we can't add edges that form triangles.
    
    # Let's try a systematic approach: 
    # For a path of length 23, to get diameter 4, we need edges that span about 5-6 steps.
    # E.g., add edges (0,5), (1,6), (2,7), ..., but need to check triangle-freeness.
    
    # Actually, let's use a randomized search with greedy improvement.
    best_set = None
    best_score = float('inf')
    
    # Try several random starting points
    random.seed(42)
    for attempt in range(50):
        # Start with a random subset of edges
        k = random.randint(4, 12)
        subset = random.sample(ALL_EDGES, k)
        current_set = list(subset)
        
        # Local search: try to improve
        improved = True
        while improved:
            improved = False
            cur_score = _score(current_set)
            
            # Try removing edges
            for i in range(len(current_set)):
                new_set = current_set[:i] + current_set[i+1:]
                if _score(new_set) < cur_score:
                    current_set = new_set
                    improved = True
                    break
            
            # Try adding edges
            if not improved:
                for e in ALL_EDGES:
                    if e not in current_set:
                        new_set = current_set + [e]
                        if _score(new_set) < cur_score:
                            current_set = new_set
                            improved = True
                            break
        
        if _score(current_set) < best_score:
            best_score = _score(current_set)
            best_set = current_set.copy()
    
    return best_set if best_set is not None else []

def _simulated_annealing():
    """Simulated annealing to find minimum-cardinality valid edge set."""
    random.seed(42)
    
    # Start with a feasible solution found by greedy
    initial = _greedy_construct()
    if initial is None or _score(initial) >= 1000:
        # If greedy didn't find valid, start with a larger random set
        initial = random.sample(ALL_EDGES, 15)
    
    current = list(initial)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    max_steps = 5000
    
    step = 0
    while T > min_T and step < max_steps:
        step += 1
        # Generate neighbor: add, remove, or swap an edge
        move = random.choice(['add', 'remove', 'swap'])
        
        if move == 'add' and len(current) < 20:
            e = random.choice(ALL_EDGES)
            if e not in current:
                new_set = current + [e]
            else:
                new_set = current
        elif move == 'remove' and len(current) > 2:
            idx = random.randint(0, len(current) - 1)
            new_set = current[:idx] + current[idx+1:]
        elif move == 'swap' and len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            e = random.choice(ALL_EDGES)
            new_set = current[:idx] + [e] + current[idx+1:]
        else:
            new_set = current
        
        new_score = _score(new_set)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current = new_set
            current_score = new_score
            if current_score < best_score:
                best = list(current)
                best_score = current_score
        
        T *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 second deadline for safety
    
    # First try greedy
    result = _greedy_construct()
    if result is not None and _score(result) < 1000:
        # Found a valid solution, try to improve with SA
        pass
    else:
        result = []
    
    # Run simulated annealing with time guard
    sa_result = _simulated_annealing()
    if sa_result is not None and _score(sa_result) < _score(result if result else []):
        result = sa_result
    
    # Final validation
    if result is not None and _is_valid_full(result):
        # Return as list of [u, v] lists
        return [[u, v] for u, v in result]
    
    # Fallback: return empty list if no valid solution found (should not happen)
    return []
# EVOLVE-BLOCK-END

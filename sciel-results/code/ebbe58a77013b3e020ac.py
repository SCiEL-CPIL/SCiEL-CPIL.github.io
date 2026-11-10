"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = defaultdict(list)
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

DEADLINE = 80.0  # safety margin before 90s kill

def has_bicolored_cycle(coloring, k):
    """Check if the subgraph induced by any two colors is a forest.
    Returns True if a bicolored cycle exists (invalid)."""
    # For each pair of colors, check if induced subgraph is a forest
    # A graph with V vertices and E edges is a forest iff E <= V - (#components)
    # Equivalently, no cycle: use union-find per color pair
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Build the subgraph induced by colors c1 and c2
            # Union-find
            parent = list(range(N))
            rank = [0] * N
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            for a, b in EDGES:
                ca, cb = coloring[a], coloring[b]
                if (ca == c1 and cb == c2) or (ca == c2 and cb == c1):
                    if not union(a, b):
                        return True  # bicolored cycle found
    return False

def is_valid(coloring, k):
    """Check adjacency constraint and bicolored cycle constraint."""
    # Adjacent vertices must differ
    for a, b in EDGES:
        if coloring[a] == coloring[b]:
            return False
    # Bicolored cycle check
    if has_bicolored_cycle(coloring, k):
        return False
    return True

def count_distinct(coloring):
    return len(set(coloring))

def initial_coloring(k, seed):
    """Create a random coloring with k colors, trying to satisfy adjacency."""
    rng = random.Random(seed)
    coloring = [0] * N
    for i in range(N):
        # Try to pick a color different from already-colored neighbors
        used = set()
        for nb in ADJ[i]:
            if nb < i:
                used.add(coloring[nb])
        available = [c for c in range(k) if c not in used]
        if available:
            coloring[i] = rng.choice(available)
        else:
            coloring[i] = rng.randint(0, k - 1)
    return coloring

def local_search(coloring, k, max_steps, seed, time_limit):
    """Simulated annealing / hill climbing to find valid coloring."""
    rng = random.Random(seed)
    best = list(coloring)
    best_valid = is_valid(best, k)
    
    T = 1.0
    cooling = 0.9995
    current = list(coloring)
    current_valid = best_valid
    
    steps = 0
    start = time.time()
    
    while steps < max_steps:
        if time.time() - start > time_limit:
            break
        
        # Pick a random vertex to recolor
        v = rng.randint(0, N - 1)
        old_c = current[v]
        new_c = rng.randint(0, k - 1)
        if new_c == old_c:
            steps += 1
            continue
        
        # Check if new color is valid for adjacency
        if any(current[nb] == new_c for nb in ADJ[v]):
            steps += 1
            T *= cooling
            continue
        
        current[v] = new_c
        new_valid = is_valid(current, k)
        
        if new_valid:
            # Accept and update best
            best = list(current)
            best_valid = True
            T = max(0.01, T * 0.99)  # cool slower when making progress
        else:
            # Decide whether to accept based on "cost"
            # Cost: number of bicolored cycles + adjacency violations
            # For simplicity, reject with high probability
            if rng.random() > 0.3 * T:
                current[v] = old_c  # revert
            # else keep the change (exploration)
        
        T *= cooling
        steps += 1
    
    if best_valid:
        return best, True
    return best, False

def try_k_colors(k, seed, time_limit):
    """Try to find a valid k-coloring."""
    # Multiple restarts
    n_restarts = max(1, int(time_limit / 0.5))
    start = time.time()
    
    for r in range(n_restarts):
        if time.time() - start > time_limit - 1:
            break
        coloring = initial_coloring(k, seed + r * 1000)
        valid = is_valid(coloring, k)
        if valid:
            return coloring
        # Local search
        coloring, valid = local_search(coloring, k, 5000, seed + r * 1000, time_limit - (time.time() - start) - 1)
        if valid:
            return coloring
    
    return None

def construct():
    start = time.time()
    deadline = DEADLINE
    
    # Try increasing numbers of colors
    # Start with small k and increase
    for k in range(2, 15):
        if time.time() > deadline - 5:
            break
        time_left = deadline - time.time() - 1
        if time_left < 2:
            break
        coloring = try_k_colors(k, 42, time_left)
        if coloring is not None:
            # Verify
            if is_valid(coloring, k):
                return [int(c) for c in coloring]
    
    # Fallback: greedy coloring with many colors
    if time.time() > deadline - 2:
        # Emergency: just return a valid coloring with many colors
        coloring = [0] * N
        for i in range(N):
            used = set()
            for nb in ADJ[i]:
                if nb < i:
                    used.add(coloring[nb])
            c = 0
            while c in used:
                c += 1
            coloring[i] = c
        # Verify and return
        k = max(coloring) + 1
        if is_valid(coloring, k):
            return [int(c) for c in coloring]
        return [int(i) for i in range(N)]
    
    # More thorough search with moderate k
    for k in range(3, 8):
        if time.time() > deadline - 3:
            break
        time_left = deadline - time.time() - 1
        if time_left < 3:
            break
        coloring = try_k_colors(k, 123 + k * 7777, time_left)
        if coloring is not None:
            if is_valid(coloring, k):
                return [int(c) for c in coloring]
    
    # Last resort: use all distinct colors (always valid)
    coloring = list(range(N))
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END

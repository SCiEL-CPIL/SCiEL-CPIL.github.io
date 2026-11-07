"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

N = P['n']
MIN_V = P['minimum_vertices']

# Build adjacency bitmask
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def distinct_degrees(vertices):
    """Compute number of distinct degrees in induced subgraph."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = 0
        for u in vertices:
            if u != v and (adj[v] >> u) & 1:
                deg += 1
        degrees.append(deg)
    return len(set(degrees))

def distinct_degrees_fast(vertices):
    """Faster version using bitmask intersection."""
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    degrees = []
    for v in vertices:
        # Count neighbors of v that are in the subset
        common = adj[v] & mask
        # Remove self if present (it shouldn't be, but safe)
        common &= ~(1 << v)
        deg = bin(common).count('1')
        degrees.append(deg)
    return len(set(degrees))

def greedy_construct(target_size, seed_rng):
    """Greedy: start empty, add vertex that maximizes distinct degrees."""
    vertices = []
    remaining = list(range(N))
    
    # Start with a random vertex
    first = seed_rng.choice(remaining)
    vertices.append(first)
    remaining.remove(first)
    
    while len(vertices) < target_size and remaining:
        best_v = None
        best_score = -1
        for v in remaining:
            test = vertices + [v]
            score = distinct_degrees_fast(test)
            if score > best_score:
                best_score = score
                best_v = v
        if best_v is not None:
            vertices.append(best_v)
            remaining.remove(best_v)
        else:
            break
    return vertices

def local_search(vertices, seed_rng, deadline):
    """Local search with add/remove/swap moves."""
    vertices = list(vertices)
    current_score = distinct_degrees_fast(vertices)
    best_vertices = list(vertices)
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        # Try swaps
        in_set = set(vertices)
        out_set = [v for v in range(N) if v not in in_set]
        
        # Random swap
        if out_set and len(vertices) >= MIN_V:
            for _ in range(20):
                if time.time() >= deadline:
                    break
                vi = seed_rng.randint(0, len(vertices) - 1)
                vo = seed_rng.choice(out_set)
                new_v = list(vertices)
                new_v[vi] = vo
                if len(set(new_v)) != len(new_v):
                    continue
                score = distinct_degrees_fast(new_v)
                if score > current_score:
                    vertices = new_v
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_vertices = list(vertices)
                    improved = True
                    in_set = set(vertices)
                    out_set = [v for v in range(N) if v not in in_set]
        
        # Try add (if below some max)
        if len(vertices) < 13 and out_set:
            for _ in range(10):
                if time.time() >= deadline:
                    break
                va = seed_rng.choice(out_set)
                new_v = vertices + [va]
                score = distinct_degrees_fast(new_v)
                if score > current_score:
                    vertices = new_v
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_vertices = list(vertices)
                    improved = True
                    in_set = set(vertices)
                    out_set = [v for v in range(N) if v not in in_set]
        
        # Try remove (if above min)
        if len(vertices) > MIN_V:
            for _ in range(5):
                if time.time() >= deadline:
                    break
                vr = seed_rng.randint(0, len(vertices) - 1)
                new_v = vertices[:vr] + vertices[vr+1:]
                if len(new_v) < MIN_V:
                    continue
                score = distinct_degrees_fast(new_v)
                if score > current_score:
                    vertices = new_v
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_vertices = list(vertices)
                    improved = True
                    in_set = set(vertices)
                    out_set = [v for v in range(N) if v not in in_set]
        
        if not improved:
            # Perturb: random restart within neighborhood
            if out_set and len(vertices) >= MIN_V:
                for _ in range(5):
                    vi = seed_rng.randint(0, len(vertices) - 1)
                    vo = seed_rng.choice(out_set)
                    new_v = list(vertices)
                    new_v[vi] = vo
                    if len(set(new_v)) != len(new_v):
                        continue
                    score = distinct_degrees_fast(new_v)
                    if score >= current_score - 1:
                        vertices = new_v
                        current_score = score
                        in_set = set(vertices)
                        out_set = [v for v in range(N) if v not in in_set]
                        break
    
    return best_vertices, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s
    
    rng = random.Random(42)
    
    best_vertices = list(range(9))  # Seed fallback
    best_score = distinct_degrees_fast(best_vertices)
    
    # Try different target sizes
    for target_size in [8, 9, 10, 11, 12]:
        if time.time() >= deadline:
            break
        for restart in range(30):
            if time.time() >= deadline:
                break
            # Greedy construction
            verts = greedy_construct(target_size, rng)
            if len(verts) < MIN_V:
                continue
            score = distinct_degrees_fast(verts)
            if score > best_score:
                best_score = score
                best_vertices = list(verts)
            
            # Local search
            if time.time() < deadline - 2:
                lv, ls = local_search(verts, rng, deadline)
                if ls > best_score:
                    best_score = ls
                    best_vertices = list(lv)
    
    # Also try some random subsets
    for _ in range(200):
        if time.time() >= deadline:
            break
        size = rng.randint(MIN_V, 12)
        verts = rng.sample(range(N), size)
        score = distinct_degrees_fast(verts)
        if score > best_score:
            best_score = score
            best_vertices = list(verts)
    
    # Final polish: try all single swaps on best
    if time.time() < deadline:
        in_set = set(best_vertices)
        out_set = [v for v in range(N) if v not in in_set]
        for i, v in enumerate(best_vertices):
            if time.time() >= deadline:
                break
            for vo in out_set:
                new_v = list(best_vertices)
                new_v[i] = vo
                if len(set(new_v)) != len(new_v):
                    continue
                score = distinct_degrees_fast(new_v)
                if score > best_score:
                    best_score = score
                    best_vertices = new_v
                    in_set = set(best_vertices)
                    out_set = [v for v in range(N) if v not in in_set]
    
    # Ensure return type is correct
    result = [int(v) for v in best_vertices]
    if len(result) < MIN_V:
        # Fallback to seed
        result = list(range(9))
    
    return result
# EVOLVE-BLOCK-END

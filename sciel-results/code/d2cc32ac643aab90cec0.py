"""Rewritten program using combinatorial search to maximize distinct induced degrees."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _induced_degrees(vertices, adj):
    """Compute degrees of each vertex in the induced subgraph."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in adj[v] if u in vset)
        degrees.append(deg)
    return degrees

def _score(vertices, adj):
    """Return number of distinct degrees in the induced subgraph."""
    if len(vertices) < P['minimum_vertices']:
        return -1
    degs = _induced_degrees(vertices, adj)
    return len(set(degs))

def _greedy_construct(adj):
    """Greedy construction: start with a random seed, add vertices that increase distinct degrees."""
    n = P['n']
    best_score = 0
    best_vertices = list(range(P['minimum_vertices']))
    
    for trial in range(50):
        random.seed(trial)
        current = list(random.sample(range(n), P['minimum_vertices']))
        cur_score = _score(current, adj)
        
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_new_score = cur_score
            
            for v in range(n):
                if v in current:
                    continue
                candidate = current + [v]
                new_score = _score(candidate, adj)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_candidate = v
            
            if best_candidate is not None:
                current.append(best_candidate)
                cur_score = best_new_score
                improved = True
        
        if cur_score > best_score:
            best_score = cur_score
            best_vertices = current[:]
        
        if best_score >= 7:  # Can't have more than 7 distinct degrees with 8 vertices
            break
    
    return best_vertices, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Phase 1: Greedy construction
    best_vertices, best_score = _greedy_construct(adj)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return [int(v) for v in best_vertices]
    
    # Phase 2: Simulated annealing / local search with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        random.seed(1000 + restart)
        current = list(random.sample(range(n), random.randint(8, 12)))
        cur_score = _score(current, adj)
        
        temperature = 2.0
        for step in range(500):
            if time.time() > deadline - 2:
                break
            
            # Random move: add, remove, or swap a vertex
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(current) < 14:
                candidates = [v for v in range(n) if v not in current]
                if not candidates:
                    continue
                v = random.choice(candidates)
                new_current = current + [v]
            elif move_type == 'remove' and len(current) > 8:
                v = random.choice(current)
                new_current = [x for x in current if x != v]
            elif move_type == 'swap' and len(current) >= 8:
                in_v = random.choice(current)
                out_candidates = [v for v in range(n) if v not in current]
                if not out_candidates:
                    continue
                out_v = random.choice(out_candidates)
                new_current = [out_v if x == in_v else x for x in current]
            else:
                continue
            
            new_score = _score(new_current, adj)
            
            if new_score >= cur_score:
                current = new_current
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_vertices = current[:]
                    if best_score >= 7:
                        return [int(v) for v in best_vertices]
            else:
                # Accept with probability based on temperature
                if random.random() < pow(2.718281828, (new_score - cur_score) / max(temperature, 0.1)):
                    current = new_current
                    cur_score = new_score
            
            temperature *= 0.995
        
        # Local polish: try all single-vertex additions/removals
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if v in current:
                    candidate = [x for x in current if x != v]
                    if len(candidate) >= 8:
                        s = _score(candidate, adj)
                        if s > cur_score:
                            current = candidate
                            cur_score = s
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_vertices = current[:]
                else:
                    candidate = current + [v]
                    s = _score(candidate, adj)
                    if s > cur_score:
                        current = candidate
                        cur_score = s
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_vertices = current[:]
            
            if best_score >= 7:
                return [int(v) for v in best_vertices]
    
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END

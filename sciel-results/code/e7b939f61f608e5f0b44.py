"""Simulated Annealing for rainbow 4-cycle edge coloring on 20 vertices."""
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    # Precompute all 4-cycles in K_20
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x
    # and the order of edges.
    # For any set of 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Note: (a,b,c,d) means edges (a,b), (b,c), (c,d), (d,a)
    
    all_cycles = []
    vertices = list(range(n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    a, b, c, d = vertices[i], vertices[j], vertices[k], vertices[l]
                    # Cycle 1: a-b-c-d-a
                    all_cycles.append(((a,b), (b,c), (c,d), (d,a)))
                    # Cycle 2: a-b-d-c-a
                    all_cycles.append(((a,b), (b,d), (d,c), (c,a)))
                    # Cycle 3: a-c-b-d-a
                    all_cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    def get_state_score(edges):
        """
        edges: dict frozenset(u,v) -> color
        returns: (num_edges, num_violations)
        """
        num_edges = len(edges)
        num_violations = 0
        for cycle in all_cycles:
            e1, e2, e3, e4 = cycle
            # Check if all edges exist
            if e1 not in edges or e2 not in edges or e3 not in edges or e4 not in edges:
                continue
            c1 = edges[e1]
            c2 = edges[e2]
            c3 = edges[e3]
            c4 = edges[e4]
            if len({c1, c2, c3, c4}) < 4:
                num_violations += 1
        return num_edges, num_violations

    # Initialize random graph
    # Start with a moderate density
    current_edges = {}
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                key = frozenset((i, j))
                current_edges[key] = random.choice(colors)
    
    best_edges = None
    best_edge_count = 0
    best_violations = 10000
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    max_steps = 100000 # Upper bound, will stop by deadline
    
    # Initial evaluation
    cur_e, cur_v = get_state_score(current_edges)
    if cur_v == 0 and cur_e > best_edge_count:
        best_edge_count = cur_e
        best_edges = dict(current_edges)
    
    step = 0
    while time.time() - start_time < DEADLINE:
        if step % 500 == 0:
            if time.time() - start_time > DEADLINE:
                break
            
        # Propose a move
        move_type = random.random()
        
        # Get a random pair
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        while u == v:
            v = random.randint(0, n-1)
        key = frozenset((u, v))
        
        new_edges = dict(current_edges) # Shallow copy
        
        if key in new_edges:
            # Edge exists
            if move_type < 0.5:
                # Remove edge
                del new_edges[key]
            else:
                # Change color
                new_edges[key] = random.choice(colors)
        else:
            # Edge does not exist
            # Add edge
            new_edges[key] = random.choice(colors)
        
        # Evaluate new state
        new_e, new_v = get_state_score(new_edges)
        
        # Objective: Maximize edges, Minimize violations
        # We use a penalty method
        penalty = 100
        cur_obj = cur_e - penalty * cur_v
        new_obj = new_e - penalty * new_v
        
        delta = new_obj - cur_obj
        
        # Accept if better, or with probability exp(delta/T) if worse
        accept = False
        if delta > 0:
            accept = True
        else:
            # Avoid overflow
            if delta / T > -50:
                prob = 2.71828 ** (delta / T)
            else:
                prob = 0
            if random.random() < prob:
                accept = True
        
        if accept:
            current_edges = new_edges
            cur_e, cur_v = new_e, new_v
            
            # Update best if valid
            if cur_v == 0 and cur_e > best_edge_count:
                best_edge_count = cur_e
                best_edges = dict(current_edges)
        
        T *= cooling
        step += 1
        
        # Early exit if we found a very dense valid graph?
        if best_edge_count >= 150:
            break

    if best_edges is None:
        # Fallback: return current state if valid, or a star
        if cur_v == 0:
            return [[min(k), max(k), v] for k, v in current_edges.items()]
        else:
            # Return a valid star
            return [[0, v, 0] for v in range(1, n)]
    
    return [[min(k), max(k), v] for k, v in best_edges.items()]

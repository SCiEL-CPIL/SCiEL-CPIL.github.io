"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    """Compute raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _initial_values():
    """Generate initial values: small numbers with some composites."""
    # Mix of small primes, composites, and a few larger numbers
    base = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20]
    random.shuffle(base)
    return base

def _initial_edges():
    """Generate initial 35 edges - prefer close-together indices for more collisions."""
    # Start with a structured set: prefer edges between nearby vertices
    edges = []
    # First add edges between consecutive and near-consecutive vertices
    for i in range(N):
        for j in range(i+1, min(i+4, N)):
            edges.append([i, j])
    random.shuffle(edges)
    edges = edges[:M]
    # Fill remaining with random
    remaining = [e for e in ALL_EDGES if e not in edges]
    random.shuffle(remaining)
    while len(edges) < M and remaining:
        edges.append(list(remaining.pop()))
    return edges

def _swap_values(values, i, j):
    """Return new values with positions i and j swapped."""
    new_vals = values[:]
    new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
    return new_vals

def _perturb_value(values, i, delta):
    """Return new values with position i perturbed by delta, keeping distinct and in range."""
    new_val = values[i] + delta
    if new_val < 1 or new_val > MAX_VAL:
        return None
    if new_val in values:
        return None
    new_vals = values[:]
    new_vals[i] = new_val
    return new_vals

def _swap_edges(edges, e1, e2):
    """Return new edges with e1 and e2 swapped (both must be valid edge pairs)."""
    new_edges = edges[:]
    # Find positions
    pos1 = None
    pos2 = None
    for idx, e in enumerate(new_edges):
        if e == e1:
            pos1 = idx
        elif e == e2:
            pos2 = idx
    if pos1 is None or pos2 is None:
        return None
    new_edges[pos1], new_edges[pos2] = new_edges[pos2], new_edges[pos1]
    return new_edges

def _add_remove_edge(edges, add_edge, remove_edge):
    """Return new edges with remove_edge removed and add_edge added."""
    new_edges = [e for e in edges if e != remove_edge]
    if add_edge in new_edges:
        return None
    new_edges.append(add_edge)
    if len(new_edges) != M:
        return None
    return new_edges

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    best_values = _initial_values()
    best_edges = _initial_edges()
    best_score = _score(best_values, best_edges)
    
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    T = 50.0
    T_min = 0.5
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Check deadline more frequently
        if time.time() > deadline - 5:
            break
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Swap two values
            i, j = random.sample(range(N), 2)
            new_values = _swap_values(current_values, i, j)
            new_edges = current_edges
        elif move_type < 0.5:
            # Perturb one value
            i = random.randint(0, N-1)
            delta = random.randint(-20, 20)
            if delta == 0:
                continue
            new_values = _perturb_value(current_values, i, delta)
            if new_values is None:
                continue
            new_edges = current_edges
        elif move_type < 0.7:
            # Swap two edges
            if len(current_edges) < 2:
                continue
            e1, e2 = random.sample(current_edges, 2)
            new_edges = _swap_edges(current_edges, e1, e2)
            if new_edges is None:
                continue
            new_values = current_values
        elif move_type < 0.85:
            # Add one edge, remove another
            remove_edge = random.choice(current_edges)
            candidates = [e for e in ALL_EDGES if e not in current_edges]
            if not candidates:
                continue
            add_edge = random.choice(candidates)
            new_edges = _add_remove_edge(current_edges, add_edge, remove_edge)
            if new_edges is None:
                continue
            new_values = current_values
        else:
            # Replace one value with a random small value
            i = random.randint(0, N-1)
            new_val = random.randint(1, 50)
            new_values = _perturb_value(current_values, i, new_val - current_values[i])
            if new_values is None:
                # Try direct assignment
                if new_val < 1 or new_val > MAX_VAL or new_val in current_values:
                    continue
                new_values = current_values[:]
                new_values[i] = new_val
            new_edges = current_edges
        
        new_score = _score(new_values, new_edges)
        
        # Acceptance criterion
        if new_score <= current_score:
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            if new_score < best_score:
                best_values = new_values[:]
                best_edges = new_edges[:]
                best_score = new_score
        else:
            # Accept worse with probability exp(-delta/T)
            import math
            delta = new_score - current_score
            if delta < 100:  # Don't accept absurdly bad moves
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
        
        T = max(T * cooling, T_min)
    
    # Return best found
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
# EVOLVE-BLOCK-END

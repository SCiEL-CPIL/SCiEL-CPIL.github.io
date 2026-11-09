"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _get_triple_index(a, b, c, n):
    """Get the linear index for triple (a,b,c) with a<b<c in lexicographic order."""
    # Number of triples with first vertex < a
    count = 0
    for i in range(a):
        count += (n - i - 1) * (n - i - 2) // 2
    # Number of triples with first vertex = a and second vertex < b
    for j in range(a + 1, b):
        count += (n - j - 1)
    # Number of triples with first vertex = a, second vertex = b, third vertex < c
    for k in range(b + 1, c):
        count += 1
    return count

def _count_violations(colors, n):
    """Count number of red K4 and blue K5 violations."""
    red_k4 = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        all_red = True
        for (x, y, z) in triples:
            idx = _get_triple_index(x, y, z, n)
            if colors[idx] != 0:
                all_red = False
                break
        if all_red:
            red_k4 += 1
    
    blue_k5 = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            x, y, z = triple
            idx = _get_triple_index(x, y, z, n)
            if colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            blue_k5 += 1
    
    return red_k4 + blue_k5

def _random_coloring(n, seed=42):
    """Generate a random initial coloring."""
    rng = random.Random(seed)
    num_triples = n * (n-1) * (n-2) // 6
    return [rng.randint(0, 1) for _ in range(num_triples)]

def _local_search(colors, n, max_iters=50000, deadline=None):
    """Local search to minimize violations using incremental updates."""
    if n < 3:
        return colors

    rng = random.Random(12345)
    best_colors = list(colors)
    num_triples = n * (n-1) * (n-2) // 6
    
    # Precompute adjacency for incremental updates
    # triple_to_quads: list of lists of (quad_tuple, other_triple_indices)
    # triple_to_quints: list of lists of (quint_tuple, other_triple_indices)
    
    # This precomputation is O(n^3 * n^2) which is fine for n<=40
    # Actually, let's just generate quads and quints on the fly if n is small, 
    # or precompute indices. 
    # For n=30, num_triples=4060. 
    # Quads: 27405. Quints: 142506.
    
    # Let's build the structures
    quad_map = {t: [] for t in range(num_triples)}
    quint_map = {t: [] for t in range(num_triples)}
    
    # Generate all quads
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        q_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        q_indices = [_get_triple_index(x, y, z, n) for (x,y,z) in q_triples]
        for i in range(4):
            others = [j for j in range(4) if j != i]
            quad_map[q_indices[i]].append((others[0], others[1], others[2]))
            
    # Generate all quints
    for quint in combinations(range(n), 5):
        q_triples = list(combinations(quint, 3))
        q_indices = [_get_triple_index(x, y, z, n) for (x,y,z) in q_triples]
        for i in range(10):
            others = [j for j in range(10) if j != i]
            quint_map[q_indices[i]].append(tuple(others))

    # Initial violation count
    red_k4_count = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        all_red = all(best_colors[_get_triple_index(x, y, z, n)] == 0 for (x,y,z) in triples)
        if all_red:
            red_k4_count += 1
            
    blue_k5_count = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            x, y, z = triple
            idx = _get_triple_index(x, y, z, n)
            if best_colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            blue_k5_count += 1
            
    current_violations = red_k4_count + blue_k5_count
    
    if deadline is not None:
        start_time = time.time()
    else:
        start_time = time.time()
        
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randint(0, num_triples - 1)
        old_color = best_colors[idx]
        new_color = 1 - old_color
        
        delta_red = 0
        delta_blue = 0
        
        # Check quads containing this triple
        for others in quad_map[idx]:
            if old_color == 0: # Was Red, becoming Blue
                # If it was a Red K4, it stops being one
                # Check if others are Red
                if all(best_colors[o] == 0 for o in others):
                    delta_red -= 1
            else: # Was Blue, becoming Red
                # If others are Red, it becomes a Red K4
                if all(best_colors[o] == 0 for o in others):
                    delta_red += 1
                    
        # Check quints containing this triple
        for others in quint_map[idx]:
            if old_color == 1: # Was Blue, becoming Red
                # If it was a Blue K5, it stops being one
                # Check if others are Blue
                if all(best_colors[o] == 1 for o in others):
                    delta_blue -= 1
            else: # Was Red, becoming Blue
                # If others are Blue, it becomes a Blue K5
                if all(best_colors[o] == 1 for o in others):
                    delta_blue += 1
                    
        new_violations = current_violations + delta_red + delta_blue
        
        if new_violations < current_violations:
            best_colors[idx] = new_color
            current_violations = new_violations
            red_k4_count += delta_red
            blue_k5_count += delta_blue
            if current_violations == 0:
                return best_colors
        # else reject, colors unchanged
    
    return best_colors

def construct():
    """Construct a coloring maximizing n with no red K4 and no blue K5."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try increasing n from small to large
    for n in range(1, 35):
        if time.time() > deadline - 5:
            break
        
        num_triples = n * (n-1) * (n-2) // 6
        if num_triples == 0:
            best_n = n
            best_colors = [0]
            continue

        # Try multiple random restarts
        found = False
        # Increase restarts and iterations for larger n
        max_restarts = 100
        max_iters = 5000
        
        for seed in range(max_restarts):
            if time.time() > deadline - 2:
                break
            
            colors = _random_coloring(n, seed)
            colors = _local_search(colors, n, max_iters=max_iters, deadline=deadline - 1)
            
            violations = _count_violations(colors, n)
            if violations == 0:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if not found and n > best_n:
            # If we couldn't find a valid coloring for this n, stop
            break
    
    # Ensure we return a valid coloring for best_n
    if best_n < 3:
        return {'n': best_n, 'colors': [0] * (best_n * (best_n-1) * (best_n-2) // 6)}
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

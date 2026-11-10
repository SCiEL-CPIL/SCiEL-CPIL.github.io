"""Edge-coloring of K_16 to maximize distinct colors with no rainbow C4."""
import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute edge indices
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1

    # Precompute all 4-cycles
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(a,d)]))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    cycles.append((edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(c,d)], edge_idx[(a,c)]))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    cycles.append((edge_idx[(a,c)], edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(a,d)]))

    num_cycles = len(cycles)

    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)

    # Precompute for each edge, the other edges in each cycle it belongs to
    edge_to_cycle_others = [dict() for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            others = tuple(e for e in cycle if e != ei)
            edge_to_cycle_others[ei][ci] = others

    def is_rainbow(c0, c1, c2, c3):
        return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

    def count_violations(colors):
        count = 0
        for cycle in cycles:
            e0, e1, e2, e3 = cycle
            if is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                count += 1
        return count

    def check_edge_change(colors, edge_i, new_color):
        """Check if changing edge_i to new_color creates any new violation.
        Returns (num_new_violations, num_removed_violations)"""
        old_color = colors[edge_i]
        if old_color == new_color:
            return 0, 0

        num_new = 0
        num_removed = 0
        for ci in edge_to_cycles[edge_i]:
            others = edge_to_cycle_others[edge_i][ci]
            o0, o1, o2 = others
            c0, c1, c2 = colors[o0], colors[o1], colors[o2]

            was_violation = is_rainbow(old_color, c0, c1, c2)
            is_violation = is_rainbow(new_color, c0, c1, c2)

            if is_violation and not was_violation:
                num_new += 1
            elif was_violation and not is_violation:
                num_removed += 1

        return num_new, num_removed

    def count_distinct_colors(colors):
        return len(set(colors))

    def local_search(colors, max_iter, rng, phase2_iters=0):
        """Hill-climbing to reduce violations, then maximize distinct colors."""
        colors = list(colors)
        violations = count_violations(colors)
        
        # Phase 1: Reduce violations to zero
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            if violations == 0:
                break
            
            # Find an edge in a violating cycle
            changed = False
            for _ in range(50):
                ei = rng.randint(0, NUM_EDGES - 1)
                if not edge_to_cycles[ei]:
                    continue
                
                # Check if this edge is in any violating cycle
                in_violation = False
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    o0, o1, o2 = others
                    if is_rainbow(colors[ei], colors[o0], colors[o1], colors[o2]):
                        in_violation = True
                        break
                
                if not in_violation:
                    continue
                
                # Try new colors that are likely to fix violations
                old_color = colors[ei]
                best_new_color = None
                best_delta = float('inf')
                
                # Try colors of neighboring edges in violating cycles
                candidate_colors = set()
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    for e in others:
                        candidate_colors.add(colors[e])
                
                # Also try a few random colors
                for _ in range(3):
                    candidate_colors.add(rng.randint(0, MAX_COLOR - 1))
                
                for nc in candidate_colors:
                    if nc == old_color:
                        continue
                    num_new, num_removed = check_edge_change(colors, ei, nc)
                    delta = num_new - num_removed
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = nc
                
                if best_new_color is not None and best_delta <= 0:
                    colors[ei] = best_new_color
                    violations += best_delta
                    changed = True
                    break
            
            if not changed:
                # Stuck - perturb
                for _ in range(5):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    colors[ei] = rng.randint(0, MAX_COLOR - 1)
                violations = count_violations(colors)

        # Phase 2: Maximize distinct colors while maintaining validity
        if violations == 0:
            used_colors = set(colors)
            for it in range(phase2_iters):
                if time.time() - start_time > DEADLINE:
                    break
                
                improved = False
                # Try to introduce new colors
                for _ in range(20):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    old_color = colors[ei]
                    
                    # Pick a color not yet used
                    available = [c for c in range(MAX_COLOR) if c not in used_colors]
                    if not available:
                        break
                    
                    nc = rng.choice(available)
                    num_new, num_removed = check_edge_change(colors, ei, nc)
                    
                    if num_new == 0:
                        colors[ei] = nc
                        used_colors.add(nc)
                        improved = True
                        break
                
                if not improved:
                    # Try changing edges to free up color slots or use new colors
                    any_change = False
                    for _ in range(10):
                        ei = rng.randint(0, NUM_EDGES - 1)
                        old_color = colors[ei]
                        nc = rng.randint(0, MAX_COLOR - 1)
                        if nc == old_color:
                            continue
                        num_new, num_removed = check_edge_change(colors, ei, nc)
                        if num_new == 0:
                            colors[ei] = nc
                            used_colors.discard(old_color)
                            used_colors.add(nc)
                            any_change = True
                            break
                    
                    if not any_change:
                        break

        return colors

    def simulated_annealing_improve(colors, deadline, rng):
        """Simulated annealing to maximize distinct colors while maintaining validity."""
        colors = list(colors)
        if count_violations(colors) > 0:
            return colors
        
        best_colors = list(colors)
        best_distinct = count_distinct_colors(colors)
        current_distinct = best_distinct
        
        T_start = 1.0
        T_end = 0.01
        total_steps = 30000
        step = 0
        
        used_colors = set(colors)
        
        while step < total_steps:
            if time.time() > deadline:
                break
            
            step += 1
            T = T_start * (T_end / T_start) ** (step / total_steps)
            
            # Pick a random edge to recolor
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Pick a new color: bias toward unused colors
            if rng.random() < 0.4:
                available = [c for c in range(MAX_COLOR) if c not in used_colors]
                if available:
                    new_color = rng.choice(available)
                else:
                    new_color = rng.randint(0, MAX_COLOR - 1)
            else:
                new_color = rng.randint(0, MAX_COLOR - 1)
            
            if new_color == old_color:
                continue
            
            # Check if this change creates any new violations
            valid = True
            for ci in edge_to_cycles[e]:
                others = edge_to_cycle_others[e][ci]
                o0, o1, o2 = others
                if is_rainbow(new_color, colors[o0], colors[o1], colors[o2]):
                    valid = False
                    break
            
            if valid:
                # Accept the move
                colors[e] = new_color
                used_colors.discard(old_color)
                used_colors.add(new_color)
                current_distinct = len(used_colors)
                
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = list(colors)
            else:
                pass
        
        return best_colors

    # Main search: multiple restarts with different strategies
    best_colors = None
    best_score = -1
    
    rng = random.Random(42)
    
    # Strategy 1: Random restarts
    for restart in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initial coloring
        colors = [rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)]
        colors = local_search(colors, 10000, rng, phase2_iters=5000)
        
        violations = count_violations(colors)
        if violations == 0:
            # Try simulated annealing improvement
            colors = simulated_annealing_improve(colors, DEADLINE, rng)
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Strategy 2: Structured initial colorings based on vertex partitions
    for restart in range(15):
        if time.time() - start_time > DEADLINE:
            break
        
        # Use a structured base coloring based on vertex mod-k partitions
        mod = rng.randint(3, 8)
        offset = rng.randint(0, MAX_COLOR - 1)
        colors = []
        for i, (u, v) in enumerate(edges):
            # Color based on (u mod mod, v mod mod)
            base = (u % mod) * mod + (v % mod)
            # Add some variation
            color = (base * 7 + offset + i * 3) % MAX_COLOR
            colors.append(color)
        
        colors = local_search(colors, 10000, rng, phase2_iters=5000)
        
        violations = count_violations(colors)
        if violations == 0:
            colors = simulated_annealing_improve(colors, DEADLINE, rng)
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Strategy 3: Greedy construction with new color bias
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        # Build coloring edge by edge, choosing colors that don't create violations
        colors = [None] * NUM_EDGES
        used_colors = set()
        
        # Order edges in a random permutation
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for ei in edge_order:
            # Find a color that doesn't create a violation
            best_color = None
            for _ in range(100):
                if rng.random() < 0.4:
                    # Try a new color
                    available = [c for c in range(MAX_COLOR) if c not in used_colors]
                    if not available:
                        nc = rng.randint(0, MAX_COLOR - 1)
                    else:
                        nc = rng.choice(available)
                else:
                    # Try a random color
                    nc = rng.randint(0, MAX_COLOR - 1)
                
                colors[ei] = nc
                # Check if this creates a violation
                creates_violation = False
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    o0, o1, o2 = others
                    if all(colors[e] is not None for e in others):
                        if is_rainbow(nc, colors[o0], colors[o1], colors[o2]):
                            creates_violation = True
                            break
                
                if not creates_violation:
                    best_color = nc
                    break
            
            if best_color is None:
                # Fallback: use a color that matches a neighbor
                best_color = 0
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    o0, o1, o2 = others
                    if colors[o0] is not None:
                        best_color = colors[o0]
                        break
                    if colors[o1] is not None:
                        best_color = colors[o1]
                        break
                    if colors[o2] is not None:
                        best_color = colors[o2]
                        break
            
            colors[ei] = best_color
            used_colors.add(best_color)
        
        # Local search to improve
        colors = local_search(colors, 8000, rng, phase2_iters=4000)
        
        violations = count_violations(colors)
        if violations == 0:
            colors = simulated_annealing_improve(colors, DEADLINE, rng)
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Strategy 4: Vector space over F_2^4 (structural approach from inspiration)
    # Assign each vertex a 4-bit vector; color edge (i,j) as a function of vi and vj
    # such that any 4-cycle has at most 3 distinct colors.
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        # Assign random 4-bit vectors to vertices
        vertex_vecs = [rng.randint(0, 15) for _ in range(N)]
        
        # Color edge (i,j) based on XOR of vertex vectors (gives 4-bit value)
        # Plus a secondary hash to spread colors
        colors = []
        for i, (u, v) in enumerate(edges):
            xor_val = vertex_vecs[u] ^ vertex_vecs[v]
            # Combine with a secondary hash for more colors
            secondary = (u * 7 + v * 3 + restart * 11) % 16
            color = (xor_val * 16 + secondary) % MAX_COLOR
            colors.append(color)
        
        colors = local_search(colors, 10000, rng, phase2_iters=5000)
        
        violations = count_violations(colors)
        if violations == 0:
            colors = simulated_annealing_improve(colors, DEADLINE, rng)
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Strategy 5: Grid-based structured coloring (from inspiration)
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        # Map vertices to 4x4 grid
        vertex_coords = [(i // 4, i % 4) for i in range(16)]
        
        colors = []
        for i, (u, v) in enumerate(edges):
            xu, yu = vertex_coords[u]
            xv, yv = vertex_coords[v]
            dx = abs(xu - xv)
            dy = abs(yu - yv)
            # Color based on (dx, dy) with some variation
            base_color = dx * 4 + dy
            variation = (u * 3 + v * 7 + restart * 5) % 20
            color = (base_color * 20 + variation) % MAX_COLOR
            colors.append(color)
        
        colors = local_search(colors, 10000, rng, phase2_iters=5000)
        
        violations = count_violations(colors)
        if violations == 0:
            colors = simulated_annealing_improve(colors, DEADLINE, rng)
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Final fallback
    if best_colors is None:
        # Use a simple valid coloring
        colors = [0] * NUM_EDGES
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 15
        
        if count_violations(colors) == 0:
            best_colors = colors
        else:
            best_colors = [0] * NUM_EDGES
    
    return best_colors

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")

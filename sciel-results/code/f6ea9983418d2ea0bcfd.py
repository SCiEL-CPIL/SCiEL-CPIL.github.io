"""Edge-coloring of K_16 to maximize distinct colors with no rainbow C4."""
import time
import random
import sys

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
    
    # Precompute all 4-cycles: for each set of 4 vertices, there are 3 distinct 4-cycles
    # A 4-cycle on vertices {a,b,c,d} (a<b<c<d) has edges:
    # Cycle 1: a-b, b-c, c-d, d-a -> edges (a,b),(b,c),(c,d),(a,d)
    # Cycle 2: a-b, b-d, d-c, c-a -> edges (a,b),(b,d),(c,d),(a,c)
    # Cycle 3: a-c, c-b, b-d, d-a -> edges (a,c),(b,c),(b,d),(a,d)
    
    cycles = []  # list of 4-tuples of edge indices
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
    
    # Precompute for each edge, which cycles it belongs to (for fast violation checking)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    def count_violations(colors):
        """Count number of rainbow 4-cycles."""
        count = 0
        for cycle in cycles:
            e0, e1, e2, e3 = cycle
            c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
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
            cycle = cycles[ci]
            # Get the other 3 edges
            others = [e for e in cycle if e != edge_i]
            o0, o1, o2 = others
            c0, c1, c2 = colors[o0], colors[o1], colors[o2]
            
            # Check if current state is a violation
            was_violation = (old_color != c0 and old_color != c1 and old_color != c2 
                           and c0 != c1 and c0 != c2 and c1 != c2)
            # Check if new state would be a violation
            is_violation = (new_color != c0 and new_color != c1 and new_color != c2
                          and c0 != c1 and c0 != c2 and c1 != c2)
            
            if is_violation and not was_violation:
                num_new += 1
            elif was_violation and not is_violation:
                num_removed += 1
        
        return num_new, num_removed
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    # Initial construction: try to build a valid coloring with many colors
    # Strategy: start with random assignment, then use local search
    
    best_colors = None
    best_score = -1
    best_violations = float('inf')
    
    # Try multiple random restarts
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initial coloring
        colors = [rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)]
        
        # Try to reduce violations using local search
        max_iter = 20000
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            violations = count_violations(colors)
            if violations == 0:
                score = count_distinct_colors(colors)
                if score > best_score:
                    best_score = score
                    best_colors = colors[:]
                    best_violations = 0
                break
            
            # Find an edge to change
            # Pick a random edge that's in a violating cycle
            changed = False
            for _ in range(20):
                ei = rng.randint(0, NUM_EDGES - 1)
                if not edge_to_cycles[ei]:
                    continue
                
                # Check if this edge is in any violating cycle
                in_violation = False
                for ci in edge_to_cycles[ei]:
                    cycle = cycles[ci]
                    e0, e1, e2, e3 = cycle
                    c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                    if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                        in_violation = True
                        break
                
                if not in_violation:
                    continue
                
                # Try new colors
                old_color = colors[ei]
                best_new_color = None
                best_delta = float('inf')
                
                # Try the colors of other edges in the same cycles (likely to fix)
                candidate_colors = set()
                for ci in edge_to_cycles[ei]:
                    cycle = cycles[ci]
                    for e in cycle:
                        if e != ei:
                            candidate_colors.add(colors[e])
                
                # Also try some random colors
                for _ in range(5):
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
                    changed = True
                    break
            
            if not changed:
                # Stuck - perturb
                for _ in range(3):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    colors[ei] = rng.randint(0, MAX_COLOR - 1)
        
        # Final check
        violations = count_violations(colors)
        if violations == 0:
            score = count_distinct_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
                best_violations = 0
        elif best_colors is None:
            # Keep best even if invalid, as fallback
            best_colors = colors[:]
            best_violations = violations
    
    # If we have a valid coloring, try to increase distinct colors
    if best_colors is not None and best_violations == 0:
        # Try to introduce new colors while maintaining validity
        used_colors = set(best_colors)
        max_iter2 = 50000
        for it in range(max_iter2):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to change a random edge to a new color
            improved = False
            for _ in range(30):
                ei = rng.randint(0, NUM_EDGES - 1)
                old_color = best_colors[ei]
                
                # Pick a color not yet used
                available = [c for c in range(MAX_COLOR) if c not in used_colors]
                if not available:
                    break
                
                nc = rng.choice(available)
                num_new, num_removed = check_edge_change(best_colors, ei, nc)
                
                if num_new == 0:
                    best_colors[ei] = nc
                    used_colors.add(nc)
                    improved = True
                    break
            
            if not improved:
                # Try changing edges that already have colors, to free up a color slot
                # Or just try random changes that don't create violations
                any_change = False
                for _ in range(10):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    old_color = best_colors[ei]
                    nc = rng.randint(0, MAX_COLOR - 1)
                    if nc == old_color:
                        continue
                    num_new, num_removed = check_edge_change(best_colors, ei, nc)
                    if num_new == 0 and nc not in used_colors:
                        best_colors[ei] = nc
                        used_colors.add(nc)
                        any_change = True
                        break
                    elif num_new == 0:
                        best_colors[ei] = nc
                        any_change = True
                        break
                
                if not any_change:
                    break
    
    # Final validation
    if best_colors is None:
        return [0] * NUM_EDGES
    
    final_violations = count_violations(best_colors)
    if final_violations > 0:
        # Try one more time with a simpler approach
        # Use a structured coloring: assign colors based on edge position
        # This guarantees no rainbow C4 if done carefully
        colors2 = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edges):
            # Color based on (i+j) mod something
            colors2[idx] = (i + j) % 15
        
        if count_violations(colors2) == 0:
            return colors2
        else:
            # Fallback: all zeros
            return [0] * NUM_EDGES
    
    return best_colors

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")

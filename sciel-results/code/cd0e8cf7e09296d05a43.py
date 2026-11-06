"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
E = N * (N - 1) // 2  # 120 edges

# EVOLVE-BLOCK-START
import random
import time

# Precompute edge index mapping
_edge_idx = {}
_eid = 0
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[(i, j)] = _eid
        _eid += 1

# Precompute cycles for each edge
# cycles_of_edge[e] = list of tuples (e1, e2, e3) such that {e, e1, e2, e3} is a 4-cycle
cycles_of_edge = [[] for _ in range(E)]

# Iterate over all 4-vertex subsets {a, b, c, d} with a < b < c < d
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                e1 = _edge_idx[(a, b)]
                e2 = _edge_idx[(b, c)]
                e3 = _edge_idx[(c, d)]
                e4 = _edge_idx[(a, d)]
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 2: a-b, b-d, c-d, a-c
                e1 = _edge_idx[(a, b)]
                e2 = _edge_idx[(b, d)]
                e3 = _edge_idx[(c, d)]
                e4 = _edge_idx[(a, c)]
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 3: a-c, b-c, b-d, a-d
                e1 = _edge_idx[(a, c)]
                e2 = _edge_idx[(b, c)]
                e3 = _edge_idx[(b, d)]
                e4 = _edge_idx[(a, d)]
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))

def is_valid_change(colors, edge, new_color):
    """
    Check if setting colors[edge] = new_color keeps all 4-cycles valid.
    Only checks cycles containing 'edge'.
    """
    for (o1, o2, o3) in cycles_of_edge[edge]:
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        # The cycle has colors: new_color, c1, c2, c3
        # Must NOT have 4 distinct colors
        if len({new_color, c1, c2, c3}) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Initial coloring: color by min(i, j)
    # This creates a valid coloring with 15 colors (0 to 14)
    colors = [0] * E
    for i in range(N):
        for j in range(i + 1, N):
            colors[_edge_idx[(i, j)]] = min(i, j)
    
    best_colors = colors[:]
    best_count = len(set(best_colors))
    
    rng = random.Random(42)
    
    # Phase 1: Greedy addition of new colors
    # Try to introduce colors 15, 16, ... up to 119
    current_count = best_count
    for target_color in range(15, 120):
        if time.time() > deadline - 10:
            break
        if target_color < current_count:
            continue
            
        found = False
        edges_to_try = list(range(E))
        rng.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if colors[e] == target_color:
                continue
            if is_valid_change(colors, e, target_color):
                colors[e] = target_color
                current_count = len(set(colors))
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
                found = True
                break
        
        if not found:
            # If we can't add this specific color, maybe we can still improve
            # by swapping or just proceed to SA
            pass
            
    # Phase 2: Simulated Annealing / Local Search
    # We want to maximize the number of distinct colors.
    # Strategy: 
    # 1. Pick a random edge.
    # 2. Pick a random new color.
    # 3. If valid, accept.
    #    - If it increases distinct count, it's a great move.
    #    - If it keeps distinct count same, accept with probability p (to explore).
    #    - If it decreases distinct count, reject (or accept with very low prob to escape local minima, 
    #      but usually for this metric, we just want to climb).
    #    
    # To escape local optima where we are stuck at K colors, we might need to allow temporary
    # decreases in count? No, the count is the objective. 
    # However, changing an edge from color A to color B (where B is already used) doesn't change count.
    # Changing from A to C (new color) increases count.
    # Changing from A to B (where A was unique) decreases count.
    
    # We will use a standard SA on the "validity" constraint, but biasing towards higher counts.
    # Actually, simpler: Just do random valid moves. Track best.
    # To encourage new colors, bias the choice of new_color towards unused colors.
    
    current = colors[:]
    current_count = len(set(current))
    
    temperature = 1.0
    cooling_rate = 0.9999
    min_temperature = 0.01
    
    while time.time() < deadline - 2:
        e = rng.randint(0, E - 1)
        old_color = current[e]
        
        # Choose new color
        # 30% chance: pick an unused color
        # 70% chance: pick any random color
        if rng.random() < 0.3:
            used = set(current)
            unused = [c for c in range(120) if c not in used]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, 119)
        else:
            new_color = rng.randint(0, 119)
            
        if new_color == old_color:
            continue
            
        if is_valid_change(current, e, new_color):
            # Calculate new count
            # We can optimize this by tracking counts, but 120 edges is small enough for set()
            # But let's try to be efficient.
            # Actually, set(current) is O(E). 120 is small.
            
            # Apply change
            current[e] = new_color
            new_count = len(set(current))
            
            # Acceptance logic
            if new_count > current_count:
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_colors = current[:]
            elif new_count == current_count:
                # Accept neutral moves to explore
                # Probability decreases with temperature? 
                # Or just always accept? 
                # If we always accept neutral moves, we might wander far from the "high density" region.
                # But since we track best, it's safe.
                # Let's accept with probability 0.5 to avoid too much wandering?
                # Or just accept.
                if rng.random() < 0.5:
                    pass # Accepted
                else:
                    current[e] = old_color
                    # Revert count? No, count is same.
            else:
                # new_count < current_count
                # Reject to avoid losing progress
                current[e] = old_color
                # Revert count? No, count is lower, we don't update current_count.
                # But we reverted the state, so current_count is still valid for the reverted state.
                
        temperature *= cooling_rate
        if temperature < min_temperature:
            # Reheat? Or just stop SA and do more greedy?
            # Let's just continue with random moves (temperature 0 effectively)
            pass
            
    return best_colors
# EVOLVE-BLOCK-END

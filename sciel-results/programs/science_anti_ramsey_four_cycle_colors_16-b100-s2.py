import random
import time

N = 16
E = N * (N - 1) // 2  # 120 edges

# Precompute edge index
_edge_idx = {}
_eid = 0
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[(i, j)] = _eid
        _eid += 1

# Precompute the three 4-cycles for each 4-vertex subset
# A 4-cycle on vertices {a,b,c,d} (sorted) has edges:
#   cycle1: (a,b),(b,c),(c,d),(d,a)
#   cycle2: (a,b),(b,d),(d,c),(c,a)
#   cycle3: (a,c),(c,b),(b,d),(d,a)
# We need: no cycle has 4 distinct edge colors.
# Precompute for each edge, the list of (other_edge1, other_edge2, other_edge3) 
# such that together they form a 4-cycle.

# For edge (i,j), find all 4-cycles containing it.
# A 4-cycle containing edge (i,j) uses two other vertices k,l.
# The cycle is i-j-k-l-i or i-j-l-k-i.
# Edges: (i,j), (j,k), (k,l), (l,i)  OR  (i,j), (j,l), (l,k), (k,i)

# For each edge e, store list of triples (e1, e2, e3) of other edges in the same 4-cycle
# When we change color of e, we need to check: for each such triple, 
#   the set {color[e], color[e1], color[e2], color[e3]} must have size <= 3.

# Actually, we need to check ALL 4-cycles, not just those containing the changed edge.
# But if we only change one edge at a time, only cycles containing that edge can become invalid.

# Precompute: for each edge e, the list of 4-cycles (as tuples of 4 edge indices) containing e.
# cycles_of_edge[e] = list of (e1, e2, e3) where {e, e1, e2, e3} forms a 4-cycle.

cycles_of_edge = [[] for _ in range(E)]

# For each 4-vertex subset {a,b,c,d} with a<b<c<d, there are 3 cycles:
# Cycle A: a-b, b-c, c-d, d-a
# Cycle B: a-b, b-d, d-c, c-a  (i.e., a-b, b-d, c-d, a-c)
# Cycle C: a-c, c-b, b-d, d-a  (i.e., a-c, b-c, b-d, a-d)

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


def count_colors(colors):
    return len(set(colors))


def is_valid_change(colors, edge, new_color):
    """Check if setting colors[edge] = new_color keeps all 4-cycles valid."""
    for (o1, o2, o3) in cycles_of_edge[edge]:
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        # The cycle has colors: new_color, c1, c2, c3
        # Must NOT have 4 distinct colors
        if len({new_color, c1, c2, c3}) == 4:
            return False
    return True


def construct():
    deadline = time.time() + 55  # safety margin
    
    # Start with a simple valid coloring: use 2 colors in a pattern
    # Actually, let's start with all zeros (1 color) and build up.
    # Better: start with a known valid low-color construction.
    
    # Strategy: Start with a valid coloring using few colors, then greedily 
    # try to recolor edges to introduce new colors.
    
    # Initial: color by vertex parity (2 colors) - always valid since any 4-cycle
    # on a bipartite-like structure... actually let's just use 1 color initially.
    colors = [0] * E
    
    # We'll try to add colors one by one.
    # For each new color c (starting from 1), try to assign it to some edge
    # such that the coloring remains valid.
    
    best = list(colors)
    best_count = 1
    
    rng = random.Random(42)
    
    # Phase 1: Greedy color addition
    # Try to introduce color 1, 2, 3, ... by finding an edge to recolor.
    max_colors_to_try = 30  # we won't exceed 120 anyway
    
    for target_color in range(1, max_colors_to_try + 1):
        if time.time() > deadline:
            break
        
        # Try many random edges to assign target_color
        found = False
        edges = list(range(E))
        rng.shuffle(edges)
        
        for e in edges:
            if colors[e] == target_color:
                continue
            if is_valid_change(colors, e, target_color):
                colors[e] = target_color
                found = True
                break
        
        if not found:
            # Try harder: allow recoloring an edge that already has a color
            # to free up a slot, or try multiple edges
            # For now, try random restarts of the greedy
            # Actually, let's try a different approach: random edge, random new color
            pass
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best = list(colors)
    
    # Phase 2: Local search - try to improve by recoloring edges
    # Use simulated annealing or hill climbing
    # Objective: maximize number of distinct colors
    
    current = list(best)
    current_count = best_count
    
    # Precompute for faster checking: for each edge, store the cycles
    # We already have cycles_of_edge.
    
    def evaluate(colors):
        return count_colors(colors)
    
    # Hill climbing with random restarts
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Randomly pick an edge and try a random new color
        e = rng.randint(0, E - 1)
        new_color = rng.randint(0, 119)
        
        if new_color == current[e]:
            continue
        
        if is_valid_change(current, e, new_color):
            old_color = current[e]
            current[e] = new_color
            new_count = count_colors(current)
            
            if new_count >= current_count:
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best = list(current)
            else:
                # Revert (since we only accept improvements for simplicity)
                current[e] = old_color
        
        # Occasionally try a "swap" or more complex move
        if step % 1000 == 0 and time.time() < deadline - 5:
            # Try to find a color that's missing and assign it
            used = set(current)
            missing = [c for c in range(120) if c not in used]
            if missing:
                target = rng.choice(missing)
                edges = list(range(E))
                rng.shuffle(edges)
                for e2 in edges[:50]:  # try first 50 random edges
                    if is_valid_change(current, e2, target):
                        current[e2] = target
                        new_count = count_colors(current)
                        if new_count > current_count:
                            current_count = new_count
                            if new_count > best_count:
                                best_count = new_count
                                best = list(current)
                        break
    
    return best


# Quick self-test
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Colors used: {len(set(result))}")
    
    # Verify
    valid = True
    colors = result
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # 3 cycles
                    for cycle in [
                        [(a,b),(b,c),(c,d),(a,d)],
                        [(a,b),(b,d),(c,d),(a,c)],
                        [(a,c),(b,c),(b,d),(a,d)],
                    ]:
                        clrs = set()
                        for (u,v) in cycle:
                            clrs.add(colors[_edge_idx[(u,v)]])
                        if len(clrs) == 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(42)
    n = 20
    num_colors = 20
    
    # Start with complete graph K20
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j, 0])
    
    # Build adjacency for quick lookup
    # edge_color[(u,v)] = color, with u < v
    def get_key(u, v):
        return (min(u,v), max(u,v))
    
    edge_color = {}
    for e in edges:
        key = get_key(e[0], e[1])
        edge_color[key] = e[2]
    
    def find_violations():
        """Find all 4-cycles with color conflicts."""
        violations = []
        # For each pair of vertices (a,b), find common neighbors
        adj = {}
        for (u, v) in edge_color:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Find all 4-cycles: a-b-c-d-a where a<b, c<d, and the cycle is valid
        for a in range(n):
            for b in range(a+1, n):
                if b not in adj.get(a, set()):
                    continue
                # Common neighbors of a and b
                common = adj.get(a, set()) & adj.get(b, set())
                for c in common:
                    if c == a or c == b:
                        continue
                    # Now find d such that d is neighbor of c and d, d != a, d != b
                    # and the 4-cycle is a-b-c-d-a
                    for d in adj.get(c, set()):
                        if d == a or d == b or d == c:
                            continue
                        if d not in adj.get(a, set()):
                            continue
                        # We have a 4-cycle: a-b-c-d-a
                        # But we need to avoid duplicates: only count if a < b and c < d or some canonical form
                        # Canonical: min(a,c) < max(a,c) and the cycle is unique
                        # Actually, let's just check all and dedupe later
                        ca = edge_color[get_key(a,b)]
                        cb = edge_color[get_key(b,c)]
                        cc = edge_color[get_key(c,d)]
                        cd = edge_color[get_key(d,a)]
                        colors = [ca, cb, cc, cd]
                        if len(set(colors)) < 4:
                            # Violation! Store the cycle and which edges to consider
                            violations.append((a, b, c, d, ca, cb, cc, cd))
        return violations
    
    def count_violations_fast():
        """Count number of violated 4-cycles."""
        adj = {}
        for (u, v) in edge_color:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                if b not in adj.get(a, set()):
                    continue
                common = adj.get(a, set()) & adj.get(b, set())
                for c in common:
                    if c == a or c == b:
                        continue
                    for d in adj.get(c, set()):
                        if d == a or d == b or d == c:
                            continue
                        if d not in adj.get(a, set()):
                            continue
                        ca = edge_color[get_key(a,b)]
                        cb = edge_color[get_key(b,c)]
                        cc = edge_color[get_key(c,d)]
                        cd = edge_color[get_key(d,a)]
                        if len(set([ca, cb, cc, cd])) < 4:
                            count += 1
        return count
    
    # Initial random coloring
    for e in edges:
        e[2] = random.randint(0, num_colors - 1)
    for e in edges:
        key = get_key(e[0], e[1])
        edge_color[key] = e[2]
    
    best_edges = [e[:] for e in edges]
    best_score = len(edges)  # all edges present, just need valid coloring
    
    # WalkSAT-style repair
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        violations = find_violations()
        if not violations:
            break
        
        # Pick a random violation
        a, b, c, d, ca, cb, cc, cd = random.choice(violations)
        
        # Try to recolor one of the four edges to fix the conflict
        # The conflicting colors are among {ca, cb, cc, cd}
        # We want to assign a new color to one edge such that it differs from the other three
        other_colors = set()
        if ca == cb or ca == cc or ca == cd:
            other_colors.add(ca)
        if cb == cc or cb == cd:
            other_colors.add(cb)
        if cc == cd:
            other_colors.add(cc)
        
        # Find a color not in the set of colors used by the other 3 edges in this cycle
        # For edge (a,b): other edges are (b,c), (c,d), (d,a) with colors cb, cc, cd
        # We want a color not in {cb, cc, cd}
        forbidden_ab = {cb, cc, cd}
        forbidden_bc = {ca, cc, cd}
        forbidden_cd = {ca, cb, cd}
        forbidden_da = {ca, cb, cc}
        
        # Randomly pick which edge to recolor
        edge_choice = random.randint(0, 3)
        if edge_choice == 0:
            key = get_key(a, b)
            forbidden = forbidden_ab
        elif edge_choice == 1:
            key = get_key(b, c)
            forbidden = forbidden_bc
        elif edge_choice == 2:
            key = get_key(c, d)
            forbidden = forbidden_cd
        else:
            key = get_key(d, a)
            forbidden = forbidden_da
        
        # Pick a random color not in forbidden
        available = [c for c in range(num_colors) if c not in forbidden]
        if available:
            new_color = random.choice(available)
            edge_color[key] = new_color
            # Update edges list
            for e in edges:
                if get_key(e[0], e[1]) == key:
                    e[2] = new_color
                    break
    
    # Final check and return
    # Verify no violations
    violations = find_violations()
    if violations:
        # Try a few more quick fixes
        for _ in range(100):
            if time.time() > deadline:
                break
            v = random.choice(violations)
            a, b, c, d, ca, cb, cc, cd = v
            edge_choice = random.randint(0, 3)
            if edge_choice == 0:
                key = get_key(a, b)
                forbidden = {cb, cc, cd}
            elif edge_choice == 1:
                key = get_key(b, c)
                forbidden = {ca, cc, cd}
            elif edge_choice == 2:
                key = get_key(c, d)
                forbidden = {ca, cb, cd}
            else:
                key = get_key(d, a)
                forbidden = {ca, cb, cc}
            available = [c for c in range(num_colors) if c not in forbidden]
            if available:
                new_color = random.choice(available)
                edge_color[key] = new_color
                for e in edges:
                    if get_key(e[0], e[1]) == key:
                        e[2] = new_color
                        break
            violations = find_violations()
            if not violations:
                break
    
    # Return the colored edges
    result = []
    for (u, v), c in edge_color.items():
        result.append([u, v, c])
    
    return result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _local_search(n, seed=42, time_limit=120.0):
    """Local search with efficient incremental violation counting for hypergraph Ramsey R(4,5;3)."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Build data structures
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_idx = {t: i for i, t in enumerate(triples)}
    
    # For each triple, which quads contain it (for red K4 tracking)
    # A quad {a,b,c,d} has 4 triples. For each triple, find the 4th vertex.
    tri_to_quads = [[] for _ in range(num_triples)]
    # For each triple (a,b,c), the quads are: (a,b,c,d) for d not in {a,b,c}
    # But we only need to track red count per quad.
    # quad_red[q] = number of red triples in quad q
    # quads: all 4-subsets
    quads = list(combinations(range(n), 4))
    num_quads = len(quads)
    quad_idx = {q: i for i, q in enumerate(quads)}
    
    # Map: for each triple, which quads contain it and what's the 4th element
    # triple (a,b,c) is in quad (a,b,c,d) for each d != a,b,c
    # We need: for quad q=(a,b,c,d), its 4 triples are (a,b,c),(a,b,d),(a,c,d),(b,c,d)
    # Build: quad_triples[qi] = list of 4 triple indices
    quad_triples = [[] for _ in range(num_quads)]
    for qi, q in enumerate(quads):
        a, b, c, d = q
        quad_triples[qi] = [triple_idx[(a,b,c)], triple_idx[(a,b,d)], triple_idx[(a,c,d)], triple_idx[(b,c,d)]]
    
    # Reverse: for each triple, list of (quad_idx, position_in_quad)
    tri_quad_membership = [[] for _ in range(num_triples)]
    for qi, q in enumerate(quads):
        for ti in quad_triples[qi]:
            tri_quad_membership[ti].append(qi)
    
    # For blue K5: for each 5-set, track number of blue triples
    quints = list(combinations(range(n), 5))
    num_quints = len(quints)
    quint_idx = {q: i for i, q in enumerate(quints)}
    
    # For each triple, which quintets contain it
    tri_quint_membership = [[] for _ in range(num_triples)]
    # For triple (a,b,c), quintets containing it: (a,b,c,d,e) for d,e > c or any combination
    # More efficiently: for each quint, its 10 triples
    quint_triples = [[] for _ in range(num_quints)]
    for qi, q in enumerate(quints):
        for tri in combinations(q, 3):
            quint_triples[qi].append(triple_idx[tri])
    
    for qi in range(num_quints):
        for ti in quint_triples[qi]:
            tri_quint_membership[ti].append(qi)
    
    # Initialize coloring: random with slight bias toward blue (since blue constraint is weaker: K5 vs K4)
    rng = random.Random(seed)
    colors = bytearray(rng.choice([0, 1]) for _ in range(num_triples))
    
    # Compute initial violation counts
    quad_red = np.zeros(num_quads, dtype=np.int8)
    for qi in range(num_quads):
        for ti in quad_triples[qi]:
            if colors[ti] == 0:
                quad_red[qi] += 1
    
    quint_blue = np.zeros(num_quints, dtype=np.int8)
    for qi in range(num_quints):
        for ti in quint_triples[qi]:
            if colors[ti] == 1:
                quint_blue[qi] += 1
    
    # Count violations
    def count_violations():
        red_viol = int(np.sum(quad_red == 4))
        blue_viol = int(np.sum(quint_blue == 10))
        return red_viol + blue_viol
    
    best_viol = count_violations()
    best_colors = colors.copy()
    
    if best_viol == 0:
        return list(best_colors)
    
    # Local search: try flipping triples that reduce violations
    # For each triple, compute the delta in violations if we flip it
    # delta_red: if we flip from blue(1) to red(0), quad_red increases by 1 for each containing quad
    #             if we flip from red(0) to blue(1), quad_red decreases by 1
    
    # Precompute for efficiency: for each triple, list of quad indices it belongs to
    # and list of quint indices
    
    steps = 0
    max_steps = 500000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Pick a random triple to try flipping
        ti = rng.randrange(num_triples)
        
        old_color = colors[ti]
        new_color = 1 - old_color
        
        # Compute delta in violations
        delta_viol = 0
        
        # Red K4 impact
        if new_color == 0:  # becoming red
            # quad_red increases by 1 for each quad containing this triple
            for qi in tri_quad_membership[ti]:
                old = quad_red[qi]
                if old == 3:
                    delta_viol += 1  # becomes a violation
                elif old == 4:
                    delta_viol -= 1  # was already a violation, stays one (but we're adding, so old was 4, new is 5? No, max is 4)
                    # Actually old==4 means it was already a violation. Adding red makes it still 4 (can't exceed 4 since only 4 triples per quad)
                    # Wait, old==4 means all 4 are already red. This triple was already red? No, we're flipping it to red, so it was blue.
                    # So old was 3 (three other reds), new becomes 4. That's a new violation.
                    # If old was 4, that means this triple was already counted as red, contradiction.
                    pass
        else:  # becoming blue (was red)
            for qi in tri_quad_membership[ti]:
                old = quad_red[qi]
                if old == 4:
                    delta_viol -= 1  # removes a violation
                elif old == 3:
                    delta_viol += 1  # wait no, removing a red from a quad with 3 reds makes it 2, no new violation
                    # old==3: was 3 reds (not a violation), becomes 2 reds. No change in violation status.
                    pass
        
        # Blue K5 impact
        if new_color == 1:  # becoming blue
            for qi in tri_quint_membership[ti]:
                old = quint_blue[qi]
                if old == 9:
                    delta_viol += 1  # becomes a violation
        else:  # becoming red (was blue)
            for qi in tri_quint_membership[ti]:
                old = quint_blue[qi]
                if old == 10:
                    delta_viol -= 1  # removes a violation
        
        # Accept if it reduces violations, or with some probability if neutral (simulated annealing lite)
        if delta_viol < 0:
            # Apply the flip
            colors[ti] = new_color
            for qi in tri_quad_membership[ti]:
                if new_color == 0:
                    quad_red[qi] += 1
                else:
                    quad_red[qi] -= 1
            for qi in tri_quint_membership[ti]:
                if new_color == 1:
                    quint_blue[qi] += 1
                else:
                    quint_blue[qi] -= 1
            best_viol += delta_viol
            if best_viol == 0:
                return list(colors)
        elif delta_viol == 0:
            # Neutral move: accept with small probability to escape local optima
            if rng.random() < 0.1:
                colors[ti] = new_color
                for qi in tri_quad_membership[ti]:
                    if new_color == 0:
                        quad_red[qi] += 1
                    else:
                        quad_red[qi] -= 1
                for qi in tri_quint_membership[ti]:
                    if new_color == 1:
                        quint_blue[qi] += 1
                    else:
                        quint_blue[qi] -= 1
        # If delta_viol > 0, reject
        
        # Periodically check if we've found a solution
        if steps % 10000 == 0:
            if best_viol == 0:
                return list(colors)
    
    # Return best found
    if best_viol == 0:
        return list(best_colors)
    return None

def _quick_valid_check(colors, n):
    """Quick check if coloring is valid."""
    triples = list(combinations(range(n), 3))
    idx = {t: i for i, t in enumerate(triples)}
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[idx[(a,b,c)]] == 0 and colors[idx[(a,b,d)]] == 0 and
            colors[idx[(a,c,d)]] == 0 and colors[idx[(b,c,d)]] == 0):
            return False
    
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True

def _fallback_coloring(n):
    """Simple modular coloring fallback."""
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Try local search for n=34
    for seed in [42, 123, 7, 999, 555, 2024]:
        if time.time() > deadline:
            break
        try:
            result = _local_search(34, seed=seed, time_limit=deadline - time.time() - 5)
            if result is not None:
                # Verify
                if _quick_valid_check(result, 34):
                    return {'n': 34, 'colors': result}
        except Exception:
            continue
    
    # Try smaller n with local search
    for n in range(33, 8, -1):
        if time.time() > deadline:
            break
        try:
            result = _local_search(n, seed=42, time_limit=deadline - time.time() - 5)
            if result is not None:
                if _quick_valid_check(result, n):
                    return {'n': n, 'colors': result}
        except Exception:
            continue
    
    # Final fallback
    n = 9
    colors = _fallback_coloring(n)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END

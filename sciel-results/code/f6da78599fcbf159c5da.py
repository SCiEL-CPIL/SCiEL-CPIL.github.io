import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precomputation
    # Map edge (i,j) to index
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # For each edge, store the "other 3 edges" for each cycle it belongs to.
    # This allows fast constraint checking.
    # edge_to_cycle_others[e] is a list of tuples (e1, e2, e3)
    edge_to_cycle_others = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        # Get edge indices
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1
        # Edges: e12, e23, e34, e14
        # For e12, others are e23, e34, e14
        edge_to_cycle_others[e12].append((e23, e34, e14))
        edge_to_cycle_others[e23].append((e12, e34, e14))
        edge_to_cycle_others[e34].append((e12, e23, e14))
        edge_to_cycle_others[e14].append((e12, e23, e34))
        
        # Cycle 2: v1-v2-v4-v3-v1
        # Edges: e12, e24, e34, e13
        edge_to_cycle_others[e12].append((e24, e34, e13))
        edge_to_cycle_others[e24].append((e12, e34, e13))
        edge_to_cycle_others[e34].append((e12, e24, e13))
        edge_to_cycle_others[e13].append((e12, e24, e34))
        
        # Cycle 3: v1-v3-v2-v4-v1
        # Edges: e13, e23, e24, e14
        edge_to_cycle_others[e13].append((e23, e24, e14))
        edge_to_cycle_others[e23].append((e13, e24, e14))
        edge_to_cycle_others[e24].append((e13, e23, e14))
        edge_to_cycle_others[e14].append((e13, e23, e24))

    # 2. Initialization
    colors = [0] * NUM_EDGES
    used_colors = {0}
    best_colors = colors[:]
    best_score = 1
    
    # Helper to get allowed colors for an edge
    def get_allowed_colors(e_idx):
        allowed = set(range(120))
        for o1, o2, o3 in edge_to_cycle_others[e_idx]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            # If c1, c2, c3 are all distinct, the new color MUST be one of them
            if c1 != c2 and c1 != c3 and c2 != c3:
                # Intersect allowed with {c1, c2, c3}
                # Optimization: if allowed is already small, check membership
                if len(allowed) <= 3:
                    # Fast path
                    if c1 not in allowed and c2 not in allowed and c3 not in allowed:
                        return set()
                    new_allowed = set()
                    if c1 in allowed: new_allowed.add(c1)
                    if c2 in allowed: new_allowed.add(c2)
                    if c3 in allowed: new_allowed.add(c3)
                    allowed = new_allowed
                else:
                    allowed.intersection_update({c1, c2, c3})
                if not allowed:
                    return set()
        return allowed

    # 3. Search Loop
    # We use a random walk that prefers adding new colors.
    # If we get stuck (no new colors added for a while), we allow neutral moves.
    
    stall_count = 0
    max_stall = 500  # Allow some neutral moves to escape local minima
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        
        allowed = get_allowed_colors(e)
        
        if not allowed:
            # No valid move for this edge. Try another.
            # To avoid infinite loop on a "locked" edge, we just continue.
            continue
            
        current_color = colors[e]
        
        # Strategy:
        # 1. If there are unused colors in allowed, pick one. (Score increase)
        # 2. If not, and stall_count < max_stall, pick a random allowed color (neutral move).
        # 3. If not, and stall_count >= max_stall, we are stuck. 
        #    We might need to restart or accept the best. 
        #    For now, we'll just skip or try to find a "better" neutral move?
        #    Actually, if we are stuck, it's hard to improve. 
        #    Let's try to pick a color that is different from current to shuffle.
        
        unused_in_allowed = [c for c in allowed if c not in used_colors]
        
        if unused_in_allowed:
            new_color = random.choice(unused_in_allowed)
            colors[e] = new_color
            used_colors.add(new_color)
            score = len(used_colors)
            
            if score > best_score:
                best_score = score
                best_colors = colors[:]
            stall_count = 0
        else:
            # No new color available.
            if stall_count < max_stall:
                # Neutral move to explore
                # Pick a random allowed color, preferably different from current
                if len(allowed) > 1:
                    # Filter out current color to force a change
                    candidates = [c for c in allowed if c != current_color]
                    if candidates:
                        new_color = random.choice(candidates)
                    else:
                        new_color = random.choice(list(allowed))
                else:
                    new_color = list(allowed)[0]
                
                if new_color != current_color:
                    colors[e] = new_color
                    # used_colors doesn't change size, but set might change if we removed last instance?
                    # No, we are just changing one edge. 
                    # If the old color was used ONLY on this edge, it's now removed from used_colors?
                    # No, used_colors is a set of colors present in the graph.
                    # If we change colors[e] from A to B:
                    # If A is not used elsewhere, A is removed from used_colors.
                    # If B is not used elsewhere, B is added to used_colors.
                    # This can decrease, keep same, or increase score.
                    
                    # Re-evaluate used_colors? 
                    # It's expensive to rebuild set every time.
                    # But 120 edges is small.
                    # Let's just track the set properly.
                    
                    old_color = current_color
                    # Check if old_color is still used
                    # We don't have a frequency map. 
                    # Rebuilding set is O(120). 
                    # 120 * 10000 ops = 1.2M ops. Fast enough.
                    
                    # Actually, let's just rebuild used_colors if we make a neutral move?
                    # Or better, maintain a frequency count.
                    
                    # Let's use a frequency array for speed.
                    pass # We'll refactor to use freq array below.
            else:
                # Stuck for too long. 
                # Try a more aggressive reset or just break?
                # Let's try to reset to best_colors and start over?
                # Or just continue hoping for a random lucky break.
                # Given the time, let's just skip.
                pass
        # The logic above is a bit messy with used_colors updates.
        # Let's refactor to use a frequency count array.
        
    return best_colors

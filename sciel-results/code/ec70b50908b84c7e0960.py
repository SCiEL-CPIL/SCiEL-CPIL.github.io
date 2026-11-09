"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n = 16
    # Precompute all K4s
    k4s = list(itertools.combinations(range(n), 4))
    num_k4 = len(k4s)  # 1820
    
    # Precompute for each edge, the list of K4 indices it belongs to
    # Edge (i,j) -> list of k4 indices
    edge_to_k4s = defaultdict(list)
    for k4_idx, k4 in enumerate(k4s):
        for a in range(4):
            for b in range(a+1, 4):
                e = (min(k4[a], k4[b]), max(k4[a], k4[b]))
                edge_to_k4s[e].append(k4_idx)
    
    # For each K4, store the 6 edges
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a in range(4):
            for b in range(a+1, 4):
                e = (min(k4[a], k4[b]), max(k4[a], k4[b]))
                edges.append(e)
        k4_edges.append(edges)
    
    # Edge index mapping
    edge_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: idx for idx, e in enumerate(edge_list)}
    num_edges = len(edge_list)  # 120
    
    # K4 edges as index lists
    k4_edge_indices = []
    for edges in k4_edges:
        k4_edge_indices.append([edge_index[e] for e in edges])
    
    # Edge to K4s as index lists
    edge_to_k4_indices = {}
    for e, k4_indices in edge_to_k4s.items():
        edge_to_k4_indices[edge_index[e]] = k4_indices
    
    def count_bad_k4s(colors):
        """Count K4s with fewer than 5 distinct colors."""
        bad = 0
        for k4_ei in k4_edge_indices:
            c = [colors[k4_ei[0]], colors[k4_ei[1]], colors[k4_ei[2]],
                 colors[k4_ei[3]], colors[k4_ei[4]], colors[k4_ei[5]]]
            if len(set(c)) < 5:
                bad += 1
        return bad
    
    def num_colors_used(colors):
        return len(set(colors))
    
    # Start from the seed: (i+j) % 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    best_colors = colors[:]
    best_bad = count_bad_k4s(best_colors)
    best_num_colors = num_colors_used(best_colors)
    
    # Target: minimize bad K4s first, then minimize colors used
    # We want bad == 0 and then minimize colors
    
    # Simulated annealing
    temp = 1.0
    cooling = 0.9999
    rng = random.Random(42)
    
    # Precompute: for each K4, the set of colors is determined by 6 edges
    # Delta cost when changing edge e from old_c to new_c:
    # For each K4 containing e, recompute the color count
    
    def delta_bad(e_idx, old_c, new_c):
        """Compute change in bad K4 count when changing edge e_idx."""
        delta = 0
        for k4_idx in edge_to_k4_indices[e_idx]:
            k4_ei = k4_edge_indices[k4_idx]
            # Get current colors for this K4
            old_set = set()
            new_set = set()
            for ei in k4_ei:
                if ei == e_idx:
                    old_set.add(old_c)
                    new_set.add(new_c)
                else:
                    c = colors[ei]
                    old_set.add(c)
                    new_set.add(c)
            old_bad = 1 if len(old_set) < 5 else 0
            new_bad = 1 if len(new_set) < 5 else 0
            delta += new_bad - old_bad
        return delta
    
    def delta_num_colors(e_idx, old_c, new_c):
        """Compute change in number of colors used."""
        # This is trickier - need to track color counts
        # For simplicity, we'll just recompute occasionally
        return 0
    
    # Better approach: track color usage counts
    def full_objective(colors):
        bad = count_bad_k4s(colors)
        nc = num_colors_used(colors)
        # Primary: minimize bad, secondary: minimize colors
        return bad * 1000 + nc
    
    current_obj = full_objective(colors)
    
    # SA loop
    step = 0
    while time.time() < deadline:
        # Pick a random edge
        e_idx = rng.randrange(num_edges)
        old_c = colors[e_idx]
        
        # Pick a new color - bias towards existing colors or slightly larger range
        # Try colors in range 0..20
        new_c = rng.randrange(0, 20)
        if new_c == old_c:
            continue
        
        # Compute delta in bad K4s
        dbad = delta_bad(e_idx, old_c, new_c)
        
        # For color count, we'll just use a simple heuristic
        # Check if new_c is already used elsewhere and old_c becomes unused
        # This is expensive to track incrementally, so we'll approximate
        # by just using bad K4 count as primary objective
        
        # Acceptance
        if dbad < 0:
            accept = True
        elif dbad == 0:
            accept = True  # neutral, always accept to explore
        else:
            # Metropolis
            import math
            if temp > 1e-10:
                accept = rng.random() < math.exp(-dbad / temp)
            else:
                accept = False
        
        if accept:
            colors[e_idx] = new_c
            current_obj += dbad * 1000  # approximate
            step += 1
            
            # Periodically check actual objective and update best
            if step % 100 == 0:
                actual_bad = count_bad_k4s(colors)
                actual_nc = num_colors_used(colors)
                if actual_bad < best_bad or (actual_bad == best_bad and actual_nc < best_num_colors):
                    best_bad = actual_bad
                    best_num_colors = actual_nc
                    best_colors = colors[:]
        
        # Cool down
        if temp > 0.01:
            temp *= cooling
        
        # If we've achieved 0 bad K4s, try to minimize colors
        if best_bad == 0:
            # Switch to color minimization mode
            # Use a different strategy: try to reduce colors
            # For now, keep going with SA to find better color counts
            pass
    
    # Final: return best colors found
    # Ensure all colors are in 0..119
    result = best_colors[:]
    # Remap colors to 0..119 if needed
    color_map = {}
    next_c = 0
    for c in result:
        if c not in color_map:
            color_map[c] = next_c
            next_c += 1
    result = [color_map[c] for c in result]
    
    return result
# EVOLVE-BLOCK-END

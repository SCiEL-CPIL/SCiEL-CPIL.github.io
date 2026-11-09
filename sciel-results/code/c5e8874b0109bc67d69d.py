import time
import random
import numpy as np
from itertools import combinations

def _color_index(i, j, n):
    """Index in the flat list for edge (i, j) with i < j."""
    return i * (i - 1) // 2 + (j - i - 1)

def _build_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _build_adj_matrices(colors, n):
    """Build adjacency matrices for both colors."""
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(2)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _color_index(i, j, n)
            c = colors[idx]
            adj[c][i][j] = 1
            adj[c][j][i] = 1
    return adj

def _total_violations(colors, n):
    """Count total monochromatic K2,2,2 subgraphs in the entire graph."""
    total = 0
    adj = _build_adj_matrices(colors, n)
    
    for c in range(2):
        for combo in combinations(range(n), 6):
            a, b, c1, d, e, f = combo
            
            pairings = [
                ((a,b),(c1,d),(e,f)),
                ((a,b),(c1,e),(d,f)),
                ((a,b),(c1,f),(d,e)),
                ((a,c1),(b,d),(e,f)),
                ((a,c1),(b,e),(d,f)),
                ((a,c1),(b,f),(d,e)),
                ((a,d),(b,c1),(e,f)),
                ((a,d),(b,e),(c1,f)),
                ((a,d),(b,f),(c1,e)),
                ((a,e),(b,c1),(d,f)),
                ((a,e),(b,d),(c1,f)),
                ((a,e),(b,f),(c,d)),
                ((a,f),(b,c1),(d,e)),
                ((a,f),(b,d),(c1,e)),
                ((a,f),(b,e),(c1,d))
            ]
            
            for p1, p2, p3 in pairings:
                valid = True
                for u in p1:
                    for w in p2:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                    for w in p3:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in p2:
                        for w in p3:
                            if adj[c][u][w] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    total += 1
    
    return total

def _is_valid_full(colors, n):
    """Check if the coloring is valid (no mono K2,2,2)."""
    return _total_violations(colors, n) == 0

def _delta_violations_flip(colors, n, idx):
    """
    Compute the change in total K2,2,2 violations when flipping edge at index idx.
    Returns (delta, old_color) where delta = new_total - old_total.
    Only considers 6-subsets containing both endpoints of the flipped edge.
    """
    # Find which edge (i,j) this index corresponds to
    # Invert the index: idx = i*(i-1)//2 + (j-i-1)
    # We need to find i and j
    i = 0
    while True:
        i2 = i * (i - 1) // 2
        if i2 > idx:
            i -= 1
            break
        i += 1
    # Now i is such that i*(i-1)//2 <= idx
    # j - i - 1 = idx - i*(i-1)//2
    j = i + 1 + (idx - i * (i - 1) // 2)
    
    old_color = colors[idx]
    new_color = 1 - old_color
    
    adj = _build_adj_matrices(colors, n)
    
    # The flipped edge (i,j) was in old_color, now in new_color
    # For a K2,2,2 to be affected, it must contain edge (i,j) in its edge set
    # And the K2,2,2 must be monochromatic in either old_color or new_color
    
    # A K2,2,2 on 6 vertices has 15 edges. For edge (i,j) to be part of a mono-K2,2,2
    # in color c, all 15 edges of that K2,2,2 must be in color c.
    # When we flip (i,j) from old_color to new_color:
    #   - Any K2,2,2 in old_color containing (i,j) is destroyed (loses one violation)
    #   - Any K2,2,2 in new_color containing (i,j) where all OTHER 14 edges are already new_color is created (gains one violation)
    
    # We need to find all 6-sets containing i and j, and all pairings where (i,j) is one of the 15 edges
    # For each such 6-set and pairing, check if the other 14 edges are all in the respective color
    
    delta = 0
    
    # Get the 4 other vertices from the remaining n-2 vertices
    other_vertices = [v for v in range(n) if v != i and v != j]
    
    for four_combo in combinations(other_vertices, 4):
        six_set = tuple(sorted([i, j] + list(four_combo)))
        # The 6 vertices are six_set. Edge (i,j) must be one of the 15 edges of the K2,2,2.
        # We need to enumerate all ways to partition the 6 vertices into 3 pairs such that (i,j) is one pair.
        # The remaining 4 vertices form 2 pairs. There are 3 ways to pair 4 vertices.
        
        remaining = list(four_combo)
        # 3 pairings of 4 vertices: (r0,r1)(r2,r3), (r0,r2)(r1,r3), (r0,r3)(r1,r2)
        pairings_4 = [
            ((remaining[0], remaining[1]), (remaining[2], remaining[3])),
            ((remaining[0], remaining[2]), (remaining[1], remaining[3])),
            ((remaining[0], remaining[3]), (remaining[1], remaining[2])),
        ]
        
        for p2, p3 in pairings_4:
            # The K2,2,2 has parts: P1={i,j}, P2=p2, P3=p3
            # Edges: all cross edges between P1-P2, P1-P3, P2-P3
            # That's 2*2 + 2*2 + 2*2 = 12 edges... wait, K2,2,2 has 12 edges, not 15!
            # K_{2,2,2} is the complete tripartite graph with parts of size 2.
            # Edges: all edges between different parts. 
            # Between P1 and P2: 2*2=4 edges
            # Between P1 and P3: 2*2=4 edges  
            # Between P2 and P3: 2*2=4 edges
            # Total: 12 edges.
            
            # So the 12 edges of this K2,2,2 are:
            # (i, p2[0]), (i, p2[1]), (j, p2[0]), (j, p2[1]),
            # (i, p3[0]), (i, p3[1]), (j, p3[0]), (j, p3[1]),
            # (p2[0], p3[0]), (p2[0], p3[1]), (p2[1], p3[0]), (p2[1], p3[1])
            
            # Edge (i,j) is NOT in this K2,2,2! Because i and j are in the same part P1.
            # In K2,2,2, there are no edges within a part.
            
            # So edge (i,j) can NEVER be part of a K2,2,2 where i and j are in the same part.
            # But i and j COULD be in different parts!
            
            # Let me reconsider. The 6 vertices can be partitioned into 3 parts of size 2 in 15 ways.
            # For edge (i,j) to be in the K2,2,2, i and j must be in DIFFERENT parts.
            # So we need to count pairings where i and j are in different parts.
            
            # Let's enumerate all 15 pairings of 6 vertices into 3 pairs of size 2.
            # Actually, the 15 pairings are the ways to partition 6 labeled vertices into 3 unordered pairs.
            # Number of such partitions = 6! / (2!^3 * 3!) = 720 / (8*6) = 15. Yes.
            
            # For edge (i,j) to be an edge in K2,2,2, i and j must be in different parts.
            # Let's just enumerate all 15 pairings of the 6-set and check if (i,j) is a cross-edge.
            
            all_vertices = list(six_set)
            # Generate all 15 perfect matchings of 6 vertices
            # We'll do it systematically
            i_pos = all_vertices.index(i)
            j_pos = all_vertices.index(j)
            
            # All ways to pair up 6 elements
            # Fix i's partner, then pair up the remaining 4
            matchings = []
            for k in range(6):
                if k == i_pos:
                    continue
                # i is paired with all_vertices[k]
                rest = [v for idx_v, v in enumerate(all_vertices) if idx_v != i_pos and idx_v != k]
                # Pair up the remaining 4: 3 ways
                r = rest
                matchings.append((all_vertices[i_pos], all_vertices[k], (r[0], r[1]), (r[2], r[3])))
                matchings.append((all_vertices[i_pos], all_vertices[k], (r[0], r[2]), (r[1], r[3])))
                matchings.append((all_vertices[i_pos], all_vertices[k], (r[0], r[3]), (r[1], r[2])))
            
            # Each matching has 3 pairs. For edge (i,j) to be a cross-edge,
            # i and j must be in different pairs.
            
            for matching in matchings:
                p1, p2, p3 = matching[1], matching[2], matching[3]
                # Check if i and j are in the same pair
                if (i in p1 and j in p1) or (i in p2 and j in p2) or (i in p3 and j in p3):
                    continue
                # i and j are in different parts, so edge (i,j) is a cross-edge
                
                # The 12 edges of this K2,2,2:
                edges = []
                for a in p1:
                    for b in p2:
                        edges.append((min(a,b), max(a,b)))
                    for b in p3:
                        edges.append((min(a,b), max(a,b)))
                for a in p2:
                    for b in p3:
                        edges.append((min(a,b), max(a,b)))
                
                # Check if (i,j) is in edges
                if (min(i,j), max(i,j)) not in edges:
                    continue  # Shouldn't happen
                
                # Now check the other 11 edges
                other_edges = [e for e in edges if e != (min(i,j), max(i,j))]
                
                # Check if all other 11 edges are in old_color (would be destroyed)
                all_old = True
                for u, v in other_edges:
                    e_idx = _color_index(u, v, n)
                    if colors[e_idx] != old_color:
                        all_old = False
                        break
                if all_old:
                    delta -= 1  # One old_color K2,2,2 destroyed
                
                # Check if all other 11 edges are in new_color (would be created)
                all_new = True
                for u, v in other_edges:
                    e_idx = _color_index(u, v, n)
                    if colors[e_idx] != new_color:
                        all_new = False
                        break
                if all_new:
                    delta += 1  # One new_color K2,2,2 created
    
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify the starting coloring is valid
    if not _is_valid_full(colors, n):
        return {'n': n, 'colors': colors}
    
    # Try to add vertex 30 using simulated annealing with incremental energy
    new_n = n + 1  # 30
    best_colors = None
    best_energy = float('inf')
    
    # Simulated annealing to find a valid 30-vertex coloring
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        # Initialize: copy 29-vertex coloring, add random edges for vertex 29
        cur_colors = colors[:]
        v_new = new_n - 1
        for i in range(v_new):
            cur_colors.append(random.randint(0, 1))
        
        # Initial energy
        cur_energy = _total_violations(cur_colors, new_n)
        
        if cur_energy == 0:
            best_colors = cur_colors[:]
            best_energy = 0
            break
        
        # Simulated annealing parameters
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.99
        max_steps = 5000
        
        best_cur_colors = cur_colors[:]
        best_cur_energy = cur_energy
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            
            # Random move: flip a random edge
            idx = random.randint(0, len(cur_colors) - 1)
            cur_colors[idx] = 1 - cur_colors[idx]
            
            # Compute delta in energy
            delta = _delta_violations_flip(cur_colors, new_n, idx)
            new_energy = cur_energy + delta
            
            # Acceptance criterion
            if new_energy < cur_energy:
                cur_energy = new_energy
                if cur_energy < best_cur_energy:
                    best_cur_energy = cur_energy
                    best_cur_colors = cur_colors[:]
                    if cur_energy == 0:
                        break
            else:
                # Accept with probability exp(-(new_energy - cur_energy) / T)
                delta_e = new_energy - cur_energy
                if delta_e > 0 and random.random() < np.exp(-delta_e / T):
                    cur_energy = new_energy
                else:
                    # Reject: undo the flip
                    cur_colors[idx] = 1 - cur_colors[idx]
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Track best solution
        if best_cur_energy < best_energy:
            best_energy = best_cur_energy
            best_colors = best_cur_colors[:]
        
        if best_energy == 0:
            break
    
    # If we found a valid 30-vertex coloring, use it
    if best_colors is not None and best_energy == 0:
        if _is_valid_full(best_colors, new_n):
            colors = best_colors
            n = new_n
    
    # Final validation
    if not _is_valid_full(colors, n):
        # Fallback: return the valid Paley 29
        colors = _build_paley(29)
        n = 29
    
    return {'n': n, 'colors': colors}

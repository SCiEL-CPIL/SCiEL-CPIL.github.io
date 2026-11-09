import time
import random
import numpy as np
from itertools import combinations

def _color_index(i, j, n):
    """Index in the flat list for edge (i, j) with i < j."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

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

def _has_K222(adj, n):
    """
    Check if adjacency matrix adj (n x n) contains K2,2,2 as a subgraph.
    K2,2,2 is the complete tripartite graph with parts of size 2.
    We need to find 6 vertices that can be partitioned into 3 pairs
    such that all edges between different pairs exist.
    """
    # Precompute: for each pair of vertices, check if they're connected
    # A K2,2,2 exists if we can find vertices a,b,c,d,e,f such that
    # all edges between {a,b} and {c,d}, {a,b} and {e,f}, {c,d} and {e,f} exist
    
    # Efficient approach: for each 6-subset, check all 15 ways to partition into 3 pairs
    # But this is O(n^6) which is too slow.
    
    # Better approach: use the fact that K2,2,2 is the complement of a 6-vertex graph
    # with specific properties. Actually, let's use a smarter method.
    
    # For each 4-subset {a,b,c,d}, check if there exist e,f such that
    # all edges between {a,b} and {c,d} exist, and all edges from {e,f} to {a,b,c,d} exist.
    
    # Actually, the cleanest approach for n<=30:
    # Enumerate all 6-subsets and check if any contains K2,2,2
    
    # Precompute complement adjacency for speed
    comp = 1 - adj
    
    for combo in combinations(range(n), 6):
        a, b, c, d, e, f = combo
        
        # Check all 15 ways to partition 6 vertices into 3 pairs
        # Pairing 1: (a,b), (c,d), (e,f)
        # Need: all edges between pair1-pair2, pair1-pair3, pair2-pair3
        pairings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in pairings:
            # Check all edges between p1 and p2
            ok = True
            for u in p1:
                for w in p2:
                    if adj[u][w] == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check all edges between p1 and p3
            for u in p1:
                for w in p3:
                    if adj[u][w] == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check all edges between p2 and p3
            for u in p2:
                for w in p3:
                    if adj[u][w] == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    
    return False

def _total_violations(colors, n):
    """Count total monochromatic K2,2,2 subgraphs in the entire graph."""
    adj = _build_adj_matrices(colors, n)
    total = 0
    for c in range(2):
        if _has_K222(adj[c], n):
            # Count how many K2,2,2 subgraphs exist
            # For energy purposes, we just need to know if any exist
            # But for better search, count them
            pass
    
    # For efficiency, just return 1 if any violation exists, else 0
    # This is sufficient for simulated annealing
    for c in range(2):
        if _has_K222(adj[c], n):
            return 1
    return 0

def _is_valid_full(colors, n):
    """Check if the coloring is valid (no mono K2,2,2)."""
    return _total_violations(colors, n) == 0

def _count_violations_detailed(colors, n):
    """Count number of monochromatic K2,2,2 subgraphs for better energy landscape."""
    adj = _build_adj_matrices(colors, n)
    total = 0
    for c in range(2):
        A = adj[c]
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
                ((a,f),(b,e),(c1,d)),
            ]
            for p1, p2, p3 in pairings:
                ok = True
                for u in p1:
                    for w in p2:
                        if A[u][w] == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for u in p1:
                    for w in p3:
                        if A[u][w] == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for u in p2:
                    for w in p3:
                        if A[u][w] == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    total += 1
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify the starting coloring is valid
    if not _is_valid_full(colors, n):
        return {'n': n, 'colors': colors}
    
    # Try to add vertex 30 using simulated annealing
    new_n = n + 1  # 30
    best_colors = None
    best_energy = float('inf')
    
    # Precompute: for the 29-vertex Paley, build adjacency
    adj29 = _build_adj_matrices(colors, n)
    
    # Simulated annealing to find a valid 30-vertex coloring
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize: copy 29-vertex coloring, add random edges for vertex 29
        cur_colors = colors[:]
        v_new = new_n - 1  # vertex index 29
        for i in range(v_new):
            cur_colors.append(random.randint(0, 1))
        
        # Build full 30-vertex adjacency
        cur_adj = _build_adj_matrices(cur_colors, new_n)
        cur_energy = _total_violations(cur_colors, new_n)
        
        if cur_energy == 0:
            best_colors = cur_colors[:]
            best_energy = 0
            break
        
        # Simulated annealing parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.998
        max_steps = 5000
        
        best_cur_colors = cur_colors[:]
        best_cur_energy = cur_energy
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            
            # Random move: flip a random edge
            idx = random.randint(0, len(cur_colors) - 1)
            old_color = cur_colors[idx]
            new_color = 1 - old_color
            cur_colors[idx] = new_color
            
            # Update adjacency matrices incrementally
            # Find which edge (i,j) this is
            # Need to reverse the index
            found = False
            for i in range(new_n):
                for j in range(i+1, new_n):
                    if _color_index(i, j, new_n) == idx:
                        cur_adj[old_color][i][j] = 0
                        cur_adj[old_color][j][i] = 0
                        cur_adj[new_color][i][j] = 1
                        cur_adj[new_color][j][i] = 1
                        found = True
                        break
                if found:
                    break
            
            # Compute new energy
            new_energy = _total_violations(cur_colors, new_n)
            
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
                delta = new_energy - cur_energy
                if delta > 0 and random.random() < np.exp(-delta / T):
                    cur_energy = new_energy
                else:
                    # Reject: undo the flip
                    cur_colors[idx] = old_color
                    cur_adj[old_color][i][j] = 1
                    cur_adj[old_color][j][i] = 1
                    cur_adj[new_color][i][j] = 0
                    cur_adj[new_color][j][i] = 0
            
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

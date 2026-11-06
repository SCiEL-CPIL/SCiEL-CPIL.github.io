import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    # We want to maximize n such that K_n has a 5-coloring with no monochromatic C4.
    # A monochromatic C4 exists iff there are two vertices u,v with >= 2 common neighbors in some color c.
    # Equivalently, for each color c, the adjacency matrix A_c must satisfy (A_c^2)[u,v] <= 1 for all u != v.
    
    # Strategy: Use simulated annealing / local search with incremental violation counting.
    # Key insight: When we change the color of edge (u,v) from c1 to c2, only violations involving
    # pairs (u,w) and (v,w) for all w != u,v can change. This gives O(n) update per move.
    
    def check_valid(n, colors):
        """Check if coloring has no monochromatic C4. O(n^3) but used only for final verification."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency matrices
        matrices = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                matrices[c][i, j] = 1
                matrices[c][j, i] = 1
                idx += 1
        
        for c in range(5):
            A = matrices[c]
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            if np.any(A2 >= 2):
                return False
        return True
    
    def solve_n(n, time_budget):
        """Find valid 5-coloring of K_n using local search with incremental updates."""
        num_edges = n * (n - 1) // 2
        if num_edges == 0:
            return []
        
        # Precompute edge list and mapping
        edges = []
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_to_idx[(i, j)] = idx
                edge_to_idx[(j, i)] = idx
                idx += 1
        
        # For each vertex, list of (neighbor, edge_index) pairs
        adj = [[] for _ in range(n)]
        for (u, v), ei in edge_to_idx.items():
            if u < v:
                adj[u].append((v, ei))
                adj[v].append((u, ei))
        
        # Initialize colors
        colors = np.random.randint(0, 5, size=num_edges)
        
        # Precompute for each color c and each pair (u,v), the count of common neighbors
        # common[c][u][v] = number of w such that edge(u,w) and edge(v,w) both have color c
        # This is expensive to compute initially: O(5 * n^3) but we can do it smarter.
        # Actually, let's compute total violations directly.
        
        def compute_violations():
            """Compute total number of (color, u, v) triples where common_neighbors >= 2."""
            total = 0
            for c in range(5):
                # Build adjacency
                A = np.zeros((n, n), dtype=np.int8)
                for (u, v), ei in edge_to_idx.items():
                    if u < v and colors[ei] == c:
                        A[u, v] = 1
                        A[v, u] = 1
                A2 = A @ A
                np.fill_diagonal(A2, 0)
                total += np.sum(A2 >= 2)
            return total
        
        # For incremental updates, we need to track:
        # For each color c and each unordered pair {u,v}, the number of common neighbors in color c.
        # Let's maintain: cc[c][u][v] for u < v
        # cc[c][u][v] = count of w (w != u, w != v) such that colors[edge(u,w)] == c and colors[edge(v,w)] == c
        
        # Initialize cc
        cc = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
        for c in range(5):
            # For each pair (u,v), count common neighbors in color c
            # This is O(n^3) per color, so O(5*n^3) total. For n=26, that's 5*26^3 ≈ 88k, fine.
            for u in range(n):
                for v in range(u + 1, n):
                    count = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        ei_uw = edge_to_idx[(min(u,w), max(u,w))]
                        ei_vw = edge_to_idx[(min(v,w), max(v,w))]
                        if colors[ei_uw] == c and colors[ei_vw] == c:
                            count += 1
                    cc[c][u][v] = count
                    cc[c][v][u] = count
        
        def count_violations_from_cc():
            """Count violations from cc array."""
            total = 0
            for c in range(5):
                total += np.sum(cc[c] >= 2)
            return total
        
        violations = count_violations_from_cc()
        best_colors = colors.copy()
        best_violations = violations
        
        # Local search with simulated annealing
        T = 1.0
        cooling = 0.9995
        max_steps = 10000000
        
        step = 0
        while step < max_steps:
            if time.time() > start_time + time_budget:
                break
            
            # Pick a random edge
            ei = random.randint(0, num_edges - 1)
            u, v = edges[ei]
            old_c = colors[ei]
            
            # Try a new color
            if random.random() < 0.8:
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    step += 1
                    continue
            else:
                new_c = (old_c + 1 + random.randint(0, 3)) % 5
            
            # Compute delta in violations when changing edge (u,v) from old_c to new_c
            # This affects cc[old_c][u][w] and cc[old_c][v][w] for all w != u,v
            # and cc[new_c][u][w] and cc[new_c][v][w] for all w != u,v
            
            delta = 0
            # For each w != u, v:
            for w in range(n):
                if w == u or w == v:
                    continue
                ei_uw = edge_to_idx[(min(u, w), max(u, w))]
                ei_vw = edge_to_idx[(min(v, w), max(v, w))]
                color_uw = colors[ei_uw]
                color_vw = colors[ei_vw]
                
                # Effect on old_c:
                # Before: edge(u,v) had color old_c
                # For pair (u,w): cc[old_c][u][w] includes w's contribution if color_uw == old_c
                #   Actually, cc[c][u][w] counts common neighbors x of u and w in color c.
                #   The edge (u,v) being color old_c contributes to cc[old_c][u][w] if color_vw == old_c? No.
                #   Let me re-think.
                #   cc[c][a][b] = number of x such that edge(a,x) has color c AND edge(b,x) has color c.
                #   When we change edge(u,v) from old_c to new_c:
                #   For pair (u,w): cc[c][u][w] is affected if the edge (u,v) or (v,w) is involved.
                #   Specifically, v is a common neighbor of u and w in color c if edge(u,v) has color c AND edge(w,v) has color c.
                #   So changing edge(u,v) from old_c to new_c:
                #   - For pair (u,w): if color_vw == old_c, then cc[old_c][u][w] decreases by 1 (v was a common neighbor, now it's not)
                #   - For pair (u,w): if color_vw == new_c, then cc[new_c][u][w] increases by 1
                #   Similarly for pair (v,w): if color_uw == old_c, cc[old_c][v][w] decreases by 1
                #   - If color_uw == new_c, cc[new_c][v][w] increases by 1
                #   Also for pair (u,v) itself: cc[c][u][v] counts common neighbors x of u and v in color c.
                #   The edge (u,v) itself doesn't contribute to cc[c][u][v] because we need x != u,v.
                #   Wait, actually cc[c][u][v] counts x such that edge(u,x)=c and edge(v,x)=c.
                #   Changing edge(u,v) doesn't affect cc[c][u][v] directly because x must be different from u and v.
                #   But it DOES affect cc[c][u][w] and cc[c][v][w] for all w.
                
                # Pair (u, w):
                if color_vw == old_c:
                    # cc[old_c][u][w] decreases by 1
                    old_val = cc[old_c][u][w]
                    new_val = old_val - 1
                    # Violation count change: if old_val >= 2, it was a violation; if new_val >= 2, still is
                    if old_val >= 2 and new_val < 2:
                        delta -= 1
                    elif old_val < 2 and new_val >= 2:
                        delta += 1
                    cc[old_c][u][w] = new_val
                    cc[old_c][w][u] = new_val
                if color_vw == new_c:
                    # cc[new_c][u][w] increases by 1
                    old_val = cc[new_c][u][w]
                    new_val = old_val + 1
                    if old_val >= 2 and new_val < 2:
                        delta -= 1
                    elif old_val < 2 and new_val >= 2:
                        delta += 1
                    cc[new_c][u][w] = new_val
                    cc[new_c][w][u] = new_val
                
                # Pair (v, w):
                if color_uw == old_c:
                    old_val = cc[old_c][v][w]
                    new_val = old_val - 1
                    if old_val >= 2 and new_val < 2:
                        delta -= 1
                    elif old_val < 2 and new_val >= 2:
                        delta += 1
                    cc[old_c][v][w] = new_val
                    cc[old_c][w][v] = new_val
                if color_uw == new_c:
                    old_val = cc[new_c][v][w]
                    new_val = old_val + 1
                    if old_val >= 2 and new_val < 2:
                        delta -= 1
                    elif old_val < 2 and new_val >= 2:
                        delta += 1
                    cc[new_c][v][w] = new_val
                    cc[new_c][w][v] = new_val
            
            # Also need to consider pair (u,v) itself? 
            # cc[c][u][v] counts x != u,v with edge(u,x)=c and edge(v,x)=c.
            # Changing edge(u,v) doesn't change this count. So no effect on cc[?][u][v].
            
            new_violations = violations + delta
            
            # Accept or reject
            if new_violations < violations:
                # Always accept improvements
                colors[ei] = new_c
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                if violations == 0:
                    break
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    colors[ei] = new_c
                    violations = new_violations
                else:
                    # Revert cc changes
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        ei_uw = edge_to_idx[(min(u, w), max(u, w))]
                        ei_vw = edge_to_idx[(min(v, w), max(v, w))]
                        color_uw = colors[ei_uw]
                        color_vw = colors[ei_vw]
                        
                        if color_vw == old_c:
                            cc[old_c][u][w] += 1
                            cc[old_c][w][u] += 1
                        if color_vw == new_c:
                            cc[new_c][u][w] -= 1
                            cc[new_c][w][u] -= 1
                        if color_uw == old_c:
                            cc[old_c][v][w] += 1
                            cc[old_c][w][v] += 1
                        if color_uw == new_c:
                            cc[new_c][v][w] -= 1
                            cc[new_c][w][v] -= 1
                    # Revert color
                    # colors[ei] is still old_c since we didn't change it
            
            T *= cooling
            if T < 0.01:
                T = 0.01
            step += 1
        
        if best_violations == 0:
            return best_colors.tolist()
        return None
    
    # Try to find the largest n
    # Start from n=26 (known to be achievable) and try to go higher
    best_n = 2
    best_colors = [0]  # n=2 fallback
    
    # Small n are trivial
    for n in range(2, 6):
        # For n <= 5, we can easily find valid colorings
        # Just use a simple pattern
        num_edges = n * (n - 1) // 2
        colors = [i % 5 for i in range(num_edges)]
        if check_valid(n, colors):
            best_n = n
            best_colors = colors
    
    # Try n from 6 up to 29, with time budget
    # Allocate more time for larger n
    remaining_time = deadline - time.time()
    
    # Try n=26 first (known achievable), then try higher
    for n in [26, 27, 28, 29, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 3:
            break
        
        if n <= best_n:
            continue
        
        # Time budget for this n
        if n >= 26:
            time_budget = min(remaining_time * 0.4, 40.0)
        elif n >= 20:
            time_budget = min(remaining_time * 0.3, 25.0)
        else:
            time_budget = min(remaining_time * 0.2, 15.0)
        
        remaining_time = deadline - time.time()
        
        # Try multiple restarts
        found = None
        num_restarts = 3 if n >= 26 else 2
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            result = solve_n(n, time_budget / num_restarts + 2.0)
            if result is not None:
                found = result
                break
        
        if found is not None:
            # Verify
            if check_valid(n, found):
                best_n = n
                best_colors = found
                # If we found n=26+, we can try to go higher
                continue
            else:
                # Shouldn't happen, but skip
                pass
    
    # Final fallback
    if best_colors is None or len(best_colors) != best_n * (best_n - 1) // 2:
        # Use n=2
        best_n = 2
        best_colors = [0]
    
    return {'n': best_n, 'colors': best_colors}

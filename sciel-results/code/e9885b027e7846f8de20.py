import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for the evaluator
    
    def get_idx(i, j, n):
        # Index for edge (i, j) in a flattened array of size n*(n-1)/2
        # i < j
        # Number of edges with first vertex < i is i*(2*n - i - 1)/2
        # Index is i*(2*n - i - 1)//2 + (j - i - 1)
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def count_violations(n, colors):
        """
        Returns the number of violating edges.
        An edge (i, j) with color c violates if |N_c(i) & N_c(j)| >= 2.
        """
        if n < 4:
            return 0
            
        # N[c][v] is a bitmask of neighbors of v connected by an edge of color c
        N = [[0] * n for _ in range(5)]
        
        # Build adjacency bitmasks
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def local_search(n, max_iters=20000, deadline=145.0):
        """
        Attempts to find a valid coloring for K_n using local search.
        Returns the colors list if found, else None.
        """
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return []
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        
        # Precompute edge list for random access
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Build initial N bitmasks
        N = [[0] * n for _ in range(5)]
        for k in range(n_edges):
            i, j = edge_list[k]
            c = colors[k]
            N[c][i] |= (1 << j)
            N[c][j] |= (1 << i)
        
        # Calculate initial violations
        violations = 0
        for k in range(n_edges):
            i, j = edge_list[k]
            c = colors[k]
            if bin(N[c][i] & N[c][j]).count('1') >= 2:
                violations += 1
        
        if violations == 0:
            return colors
            
        # Local Search Loop
        # We prioritize fixing violations.
        # We can maintain a list of violating edges, but rebuilding it is O(E).
        # Instead, we pick random edges. If an edge is violating, we try to fix it.
        # If not, we skip.
        
        # To be more effective, we can keep a set of violating edges.
        # However, updating the set is complex. 
        # A simpler approach: 
        # 1. Pick random edge.
        # 2. Check if it violates.
        # 3. If yes, try to recolor to a color that doesn't violate.
        # 4. If no, do nothing (or maybe small perturbation, but let's stick to greedy repair first).
        
        # Optimization: If we get stuck, we can randomize.
        # But simple greedy repair often works for small n.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            k = random.randint(0, n_edges - 1)
            i, j = edge_list[k]
            c = colors[k]
            
            # Check if this edge is currently violating
            common = N[c][i] & N[c][j]
            if bin(common).count('1') < 2:
                # Not a violation, skip. 
                # To avoid getting stuck in a local minimum of "non-violating but not global", 
                # we might need to accept moves that increase violations temporarily? 
                # No, the goal is to reach 0. Greedy reduction is standard.
                # If we have many violations, random picking will hit them eventually.
                # If we have few, it might take time.
                # Let's try to be smarter: keep a list of violating indices?
                # For n=12, E=66. It's small. We can just scan or keep a list.
                # Let's keep a list of violating edges for efficiency.
                pass
            
            # Actually, maintaining a list of violations is better.
            # Let's restart the logic with a violation list.
            # For simplicity and speed in this specific implementation, 
            # scanning all edges to find a violation is O(E) per step. 
            # With E=66, this is fast.
            
            # Let's refine: Find ANY violating edge.
            # If none, return success.
            # If yes, pick one randomly from the violating set? 
            # Or just pick the first one? Random is better to avoid cycles.
            
            # Let's implement a version that maintains a list of violating edges.
            # It's more complex to update. 
            # Alternative: Just pick random edge. If it violates, fix. 
            # If not, do nothing.
            # This converges slowly if violations are rare.
            
            # Let's use a hybrid: 
            # 1. Maintain `violating_edges` list.
            # 2. When an edge changes, update `violating_edges`.
            
            # This is getting complicated. Let's stick to the simple random pick 
            # but increase iterations. For n=12, it should be fast enough.
            
            # Let's try a different strategy: 
            # Pick a random edge. 
            # If it's violating, try all 5 colors. 
            # If a color reduces total violations (or just makes this edge non-violating without creating new ones on that edge's endpoints? No, it affects neighbors), pick it.
            
            # Simpler: 
            # If edge (i,j) is violating with color c:
            # Try new_c in {0..4} \ {c}
            # Check if (i,j) with new_c is safe (no C4 on this edge).
            # If safe, apply it.
            
            if bin(N[c][i] & N[c][j]).count('1') >= 2:
                # It is violating. Try to fix.
                fixed = False
                for new_c in random.sample(range(5), 5):
                    if new_c == c:
                        continue
                    # Check if new_c is safe for edge (i,j)
                    # Safe if |N[new_c][i] & N[new_c][j]| < 2
                    if bin(N[new_c][i] & N[new_c][j]).count('1') < 2:
                        # Apply change
                        # Remove from N[c]
                        N[c][i] &= ~(1 << j)
                        N[c][j] &= ~(1 << i)
                        # Add to N[new_c]
                        N[new_c][i] |= (1 << j)
                        N[new_c][j] |= (1 << i)
                        
                        colors[k] = new_c
                        fixed = True
                        break
                
                if not fixed:
                    # No single color change fixes this edge locally.
                    # This can happen. We might need to move a neighbor.
                    # For now, skip.
                    pass

        # Final check
        if count_violations(n, colors) == 0:
            return colors
        return None

    best_n = 11
    # Seed colors for n=11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n
    # We know n=11 is valid.
    # Try n=12, 13, etc.
    
    for n in range(12, 21):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use local search
            res = local_search(n, max_iters=5000, deadline=deadline)
            if res is not None:
                best_n = n
                best_colors = res
                found = True
                break
        
        # If we found a valid coloring for n, we can try to extend to n+1?
        # Or just continue the loop which tries n+1 from scratch.
        # From scratch is safer and often fast for these small n.
        
    return {'n': best_n, 'colors': best_colors}

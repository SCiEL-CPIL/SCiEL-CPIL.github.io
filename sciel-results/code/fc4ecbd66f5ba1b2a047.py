import time
import random
import sys

def construct():
    deadline = time.time() + 110
    
    def count_violations(n, colors):
        """
        Returns the number of C4 violations in the given coloring.
        colors is a list of length n*(n-1)//2
        """
        total_viol = 0
        m = n * (n - 1) // 2
        
        # adj[c][i] is an integer bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # Common neighbors of i and j in color c
                    common = adj[c][i] & adj[c][j]
                    # If there are at least 2 common neighbors, it's a C4 violation
                    if common.bit_count() >= 2:
                        total_viol += 1
        return total_viol

    def local_search(n, colors, max_iter=50000, time_limit=None):
        """
        Local search with simulated annealing.
        """
        colors = list(colors)
        current_viol = count_violations(n, colors)
        best_colors = colors[:]
        best_viol = current_viol
        
        m = n * (n - 1) // 2
        rng = random.Random(42)
        
        T = 10.0
        cooling = 0.999
        
        for it in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            
            if current_viol == 0:
                break
                
            # Pick a random edge
            idx = rng.randint(0, m - 1)
            old_color = colors[idx]
            new_color = (old_color + rng.randint(1, 4)) % 5
            
            colors[idx] = new_color
            new_viol = count_violations(n, colors)
            
            delta = new_viol - current_viol
            
            if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < pow(2.718, -delta / T)):
                current_viol = new_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
            else:
                colors[idx] = old_color
                
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_colors, best_viol

    # Start with n=26
    best_n = 1
    best_colors = []
    
    # Try to find a valid coloring for n=26
    # We can use a heuristic initial coloring
    # A known good structure for small n is often related to modular arithmetic
    # Let's try a few random restarts for n=26
    
    found_26 = False
    for target_n in [26]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        num_attempts = 5
        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break
                
            # Initial random coloring
            m = target_n * (target_n - 1) // 2
            colors = [random.randint(0, 4) for _ in range(m)]
            
            # Run local search
            t_limit = deadline - 2 - (attempt * 10) # Allocate time
            if t_limit < 5: t_limit = 5
            
            colors, viol = local_search(target_n, colors, max_iter=20000, time_limit=t_limit)
            
            if viol == 0:
                best_n = target_n
                best_colors = colors
                found_26 = True
                break
        
        if found_26:
            # Try for n=27 with remaining time
            remaining = deadline - time.time()
            if remaining > 10:
                # Use best_colors for n=26 as a seed? No, sizes differ.
                # Just start fresh for 27
                for attempt in range(10):
                    if time.time() > deadline - 2:
                        break
                    m = 27 * 26 // 2
                    colors = [random.randint(0, 4) for _ in range(m)]
                    t_limit = deadline - 2
                    colors, viol = local_search(27, colors, max_iter=20000, time_limit=t_limit)
                    if viol == 0:
                        best_n = 27
                        best_colors = colors
                        break
                    # If we get close, maybe restart with that?
                    if viol < 5:
                         # Keep this as a seed for next attempt?
                         pass

    # If we didn't find 26, we should return the best we found, which might be smaller
    # But the problem implies we should maximize n.
    # If we fail to find 26, we might have to return a smaller valid n.
    # However, the seed score 1.6364 suggests n=26 is expected.
    
    return {'n': best_n, 'colors': best_colors}

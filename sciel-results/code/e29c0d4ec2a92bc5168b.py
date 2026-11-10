"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 16
    target_colors = 13
    
    # Try to find a valid coloring with target_colors colors
    # using a structured approach
    
    # Strategy: Use a 13-coloring based on vertex partition
    # Partition 16 vertices into groups and use structured coloring
    
    # Let's try: color(i,j) based on (i-j) mod 13, but adjusted
    # Actually, let's use a known construction
    
    # Construction: Use 13 colors via a 13-vertex structure extended
    # For vertices 0..12: color(i,j) = (i-j) mod 13 (using colors 0..12)
    # For vertices 13,14,15: assign colors carefully
    
    # Better: use a cyclic group Z_13 plus 3 extra vertices
    # This is complex. Let me try a direct backtracking with pruning.
    
    # Given time constraints, let's try a heuristic construction
    # that is likely to work with 13 or 14 colors.
    
    # Approach: Use 14 colors with a structured pattern
    # Partition into 4 groups of 4: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}
    
    # Color scheme:
    # Within group: use 3 colors (0,1,2) based on position
    # Between groups: use 11 colors (3..13) based on group pair and positions
    
    # Actually, let me try a simpler proven approach
    # Use color = (i * 13 + j) % 14 or similar
    
    # Let me try: color(i,j) = (i - j) % 14 for i < j, but this uses 13 colors (0..12) since (i-j)%14 for i<j gives 1..13
    # Wait, (i-j) mod 14 where i<j gives values 1..13, so 13 distinct values
    
    # Check: for any 4 vertices a<b<c<d, the 6 edges have colors:
    # (b-a)%14, (c-a)%14, (d-a)%14, (c-b)%14, (d-b)%14, (d-c)%14
    # These are 6 values in {1,...,13}. Need at least 5 distinct.
    
    # Counterexample: a=0,b=1,c=2,d=3: colors 1,2,3,1,2,1 -> {1,2,3} = 3 colors. FAILS.
    
    # So simple modular doesn't work.
    
    # Let me try a different approach: use a Latin square based coloring
    # Or use the fact that we need a "5-colorful" K4 constraint
    
    # Given the complexity, let me use a randomized local search
    # to find a valid 13-coloring or 14-coloring
    
    random.seed(42)
    
    def count_colors_in_k4(colors, v1, v2, v3, v4):
        """Count distinct colors in the 6 edges of K4 on vertices v1,v2,v3,v4"""
        edges = [(v1,v2), (v1,v3), (v1,v4), (v2,v3), (v2,v4), (v3,v4)]
        cset = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            cset.add(colors[u * n + v])
        return len(cset)
    
    def check_validity(colors):
        """Check if every K4 has at least 5 distinct colors"""
        for v1 in range(n):
            for v2 in range(v1+1, n):
                for v3 in range(v2+1, n):
                    for v4 in range(v3+1, n):
                        if count_colors_in_k4(colors, v1, v2, v3, v4) < 5:
                            return False
        return True
    
    # Use local search: start with random coloring, flip edges to reduce violations
    # Objective: minimize number of K4s with < 5 colors
    
    def count_violations(colors):
        count = 0
        for v1 in range(n):
            for v2 in range(v1+1, n):
                for v3 in range(v2+1, n):
                    for v4 in range(v3+1, n):
                        if count_colors_in_k4(colors, v1, v2, v3, v4) < 5:
                            count += 1
        return count
    
    # Start with a structured initial coloring
    # Use 13 colors in a pattern that might work better
    colors = [0] * (n * n)
    
    # Initial: use (i+j) % 13
    for i in range(n):
        for j in range(i+1, n):
            colors[i * n + j] = (i + j) % 13
            colors[j * n + i] = colors[i * n + j]
    
    # Local search
    best_colors = colors[:]
    best_violations = count_violations(best_colors)
    
    import time
    start_time = time.time()
    max_time = 100  # seconds
    
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old_color = colors[i * n + j]
        new_color = random.randint(0, 12)
        
        if old_color == new_color:
            continue
        
        # Compute change in violations
        # Only K4s containing edge (i,j) are affected
        # A K4 containing (i,j) has 2 other vertices from the remaining 14
        old_violations = 0
        new_violations = 0
        
        for k in range(n):
            if k == i or k == j:
                continue
            for l in range(k+1, n):
                if l == i or l == j:
                    continue
                # K4 on {i, j, k, l}
                # Count violations before and after
                # Before:
                c1 = colors[i*n+j]
                c2 = colors[i*n+k] if i<k else colors[k*n+i]
                c3 = colors[i*n+l] if i<l else colors[l*n+i]
                c4 = colors[j*n+k] if j<k else colors[k*n+j]
                c5 = colors[j*n+l] if j<l else colors[l*n+j]
                c6 = colors[k*n+l] if k<l else colors[l*n+k]
                old_set = len({c1, c2, c3, c4, c5, c6})
                
                # After:
                c1_new = new_color
                new_set = len({c1_new, c2, c3, c4, c5, c6})
                
                if old_set < 5:
                    old_violations += 1
                if new_set < 5:
                    new_violations += 1
        
        delta = new_violations - old_violations
        
        if delta <= 0:
            colors[i * n + j] = new_color
            colors[j * n + i] = new_color
            best_violations += delta
            if best_violations == 0:
                break
        else:
            # Sometimes accept worse moves to escape local minima
            if random.random() < 0.1:
                colors[i * n + j] = new_color
                colors[j * n + i] = new_color
                best_violations += delta
    
    # Convert to output format
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(colors[i * n + j])
    
    return result
# EVOLVE-BLOCK-END


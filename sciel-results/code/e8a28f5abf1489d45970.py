import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Generate initial K32 coloring (Greenwood-Gleason / Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_initial_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    # Helper: get index in flat list for edge (i, j) with i < j
    def edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Helper: decode edge index to (i, j)
    def idx_to_edge(e, n):
        # Solve for i: sum_{k=0}^{i-1} (n-1-k) <= e < sum_{k=0}^{i} (n-1-k)
        # Approximation: i(n-1) - i(i-1)/2 ~ e
        # i ~ (n-1) - sqrt((n-1)^2 - 2e)
        i = int((n - 1) - ((n - 1)**2 - 2*e)**0.5)
        # Adjust i to be correct
        while edge_idx(i, i+1, n) > e:
            i -= 1
        while edge_idx(i+1, i+2, n) <= e:
            i += 1
        # Now find j
        # e = i(n-1) - i(i-1)/2 + (j-i-1)
        # j - i - 1 = e - (i(n-1) - i(i-1)/2)
        rem = e - (i * (n - 1) - i * (i - 1) // 2)
        j = i + 1 + rem
        return i, j

    # Helper: compute threats for a given coloring and n
    def compute_threats(colors, n):
        threats = [0] * (n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j, n)]
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ik_idx = edge_idx(i, k, n)
                    jk_idx = edge_idx(j, k, n)
                    if colors[ik_idx] == c and colors[jk_idx] == c:
                        count += 1
                threats[edge_idx(i, j, n)] = count
        return threats

    def get_total_threats_delta(i, j, c_new, colors, threats, n):
        idx = edge_idx(i, j, n)
        c_old = colors[idx]
        if c_old == c_new:
            return 0
        
        # Current threats for this edge are removed
        delta = -threats[idx]
        
        # New threats for this edge are added
        new_threats_count = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            if colors[ik_idx] == c_new and colors[jk_idx] == c_new:
                new_threats_count += 1
                # This new threat also affects neighbors
                # But for TOTAL threats, we just count the triangle once?
                # No, "Total Threats" as defined by sum(threats[e]) counts each triangle 3 times.
                # Triangle (i, j, k) contributes 1 to threats[(i, j)], 1 to threats[(i, k)], 1 to threats[(j, k)].
                # So if we form a new triangle, total threats increases by 3.
                # If we break an old triangle, total threats decreases by 3.
                
                # Wait, my `delta` calculation above only accounted for the edge (i, j).
                # I need to account for the other two edges.
                
                # Let's re-evaluate.
                # Old state: Edge (i, j) had color c_old.
                # If c_old formed a triangle with k, it contributed 1 to threats[(i, j)].
                # And it contributed 1 to threats[(i, k)] and 1 to threats[(j, k)].
                # So removing c_old removes 3 from total threats (if it was a triangle).
                
                # New state: Edge (i, j) has color c_new.
                # If c_new forms a triangle with k, it adds 1 to threats[(i, j)], 1 to threats[(i, k)], 1 to threats[(j, k)].
                # So adding c_new adds 3 to total threats (if it forms a triangle).
                
                # So delta_total = -3 * (old_triangles_broken) + 3 * (new_triangles_formed).
                
                # How many old triangles are broken?
                # For each k, if colors[ik_idx]==c_old and colors[jk_idx]==c_old, then (i, j, k) was a triangle.
                # How many new triangles are formed?
                # For each k, if colors[ik_idx]==c_new and colors[jk_idx]==c_new, then (i, j, k) is a new triangle.
                
                pass
        
        # Let's implement this correctly.
        old_tri = 0
        new_tri = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            c_ik = colors[ik_idx]
            c_jk = colors[jk_idx]
            
            if c_ik == c_old and c_jk == c_old:
                old_tri += 1
            if c_ik == c_new and c_jk == c_new:
                new_tri += 1
        
        return 3 * (new_tri - old_tri)

    # Revised Strategy: Simulated Annealing on Total Threats
    # State: colors list.
    # Energy: Total Threats (sum of threats array).
    # Move: Flip a random edge to a random color.
    # Delta Energy: Change in total threats.
    
    def try_extend(n, colors, threats):
        """
        Try to extend K_n to K_{n+1}.
        Returns (success, new_colors, new_threats)
        """
        new_n = n + 1
        # New colors list
        # Old edges are at the same indices?
        # No, the index formula depends on N.
        # We need to map old colors to new indices.
        
        # This is a pain.
        # Alternative: Keep the 2D matrix representation internally?
        # Or just rebuild the flat list.
        
        # Let's rebuild.
        # Old N = n. New N = n+1.
        # Edges (i, j) with i, j < n:
        # Old idx: i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        # New idx: i * (new_n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        new_colors = [0] * (new_n * (new_n - 1) // 2)
        
        # Copy old edges
        for i in range(n):
            for j in range(i + 1, n):
                old_idx = edge_idx(i, j, n)
                new_idx = edge_idx(i, j, new_n)
                new_colors[new_idx] = colors[old_idx]
        
        # Initialize new edges (n, k) for k < n randomly
        for k in range(n):
            new_idx = edge_idx(k, n, new_n)
            new_colors[new_idx] = random.randint(0, 3)
        
        # Compute threats for new coloring
        new_threats = compute_threats(new_colors, new_n)
        
        # Now we want to minimize threats to 0.
        # We only care about edges involving the new vertex n?
        # No, changing an old edge might help.
        # But changing old edges is risky (might break validity of old part).
        # However, if we use SA, we can allow temporary invalidity.
        
        # Let's run a short SA to minimize threats.
        # Temperature starts high, cools down.
        # We want to reach 0 threats.
        
        current_energy = sum(new_threats)
        T = 1.0
        T_min = 0.01
        steps = 2000
        step_size = 1.0 / steps
        
        for s in range(steps):
            if time.time() > deadline - 5:
                break
            if current_energy == 0:
                return True, new_colors, new_threats
            
            # Pick random edge
            e = random.randint(0, len(new_colors) - 1)
            # Decode e to (i, j)
            # This is slow. Let's precompute a list of (i, j) for the current N?
            # Or just iterate.
            # For N=50, 1225 edges.
            # Decoding:
            # i = 0
            # remaining = e
            # while remaining >= new_n - 1 - i:
            #    remaining -= new_n - 1 - i
            #    i += 1
            # j = i + 1 + remaining
            
            i = 0
            rem = e
            while rem >= new_n - 1 - i:
                rem -= new_n - 1 - i
                i += 1
            j = i + 1 + rem
            
            c_old = new_colors[e]
            c_new = random.randint(0, 3)
            if c_old == c_new:
                continue
                
            delta = get_total_threats_delta(i, j, c_new, new_colors, new_threats, new_n)
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                new_colors[e] = c_new
                # Update threats
                # We need to update new_threats array.
                # This is the same incremental update problem.
                # Let's just recompute threats every 100 steps?
                # Or use the delta to update the SUM, but we need the array for future deltas.
                # If we don't update the array, future deltas are wrong.
                
                # Let's recompute threats every 50 steps.
                if s % 50 == 0:
                    new_threats = compute_threats(new_colors, new_n)
                    current_energy = sum(new_threats)
                else:
                    # Approximate energy update
                    current_energy += delta
                    # This is dangerous if array is stale.
                    # But for SA, it might be okay if we recompute periodically.
                    
            T -= step_size
            if T < T_min:
                T = T_min
        
        if sum(new_threats) == 0:
            return True, new_colors, new_threats
        
        return False, new_colors, new_threats

    # Main Loop
    current_n = 32
    current_colors = get_initial_32()
    best_n = 32
    best_colors = current_colors[:]
    current_threats = compute_threats(current_colors, current_n)
    
    while current_n < 62 and time.time() < deadline - 10:
        success, new_colors, new_threats = try_extend(current_n, current_colors, current_threats)
        if success:
            current_n += 1
            current_colors = new_colors
            current_threats = new_threats
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
        else:
            # Failed to extend.
            # Try to perturb current coloring to escape local minimum.
            # Run SA on current_n to minimize threats (should be 0).
            # If it finds a different valid coloring, maybe it's easier to extend.
            
            # Let's just restart the extension with a different random seed?
            # Or perturb the new vertex edges.
            
            # For now, just break and return best found.
            # To improve, we could try multiple random initializations for the new vertex.
            
            # Let's try a few more times with different random initializations for the new vertex.
            found = False
            for attempt in range(10):
                # Re-initialize new vertex edges randomly
                # This is done inside try_extend.
                # Let's just call try_extend again.
                success, new_colors, new_threats = try_extend(current_n, current_colors, current_threats)
                if success:
                    current_n += 1
                    current_colors = new_colors
                    current_threats = new_threats
                    if current_n > best_n:
                        best_n = current_n
                        best_colors = current_colors[:]
                    found = True
                    break
                if time.time() > deadline - 5:
                    break
            
            if not found:
                # Stuck.
                # Try to perturb the existing coloring.
                # We can do a random walk on valid colorings of K_current_n.
                # This is complex.
                # Let's just break.
                break

    return {'n': best_n, 'colors': best_colors}

# Test
# print(construct())

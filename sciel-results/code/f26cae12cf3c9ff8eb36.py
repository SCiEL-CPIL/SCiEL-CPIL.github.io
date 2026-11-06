import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    n = 16
    num_edges = 120
    num_k4s = 1820
    
    # Precompute edge indices
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1
            
    # Precompute K4s and their edges
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_edges.append(edges)
        k4_id = len(k4_edges) - 1
        for e in edges:
            edge_to_k4s[e].append(k4_id)
            
    def get_1factorization_colors():
        """
        Returns a list of 120 colors based on a 1-factorization of K16.
        Uses 15 colors (0..14).
        Vertex 15 is the center. Vertices 0..14 are on a cycle.
        Factor i (i=0..14) consists of:
        - Edge (15, i)
        - Edges (i+j mod 15, i-j mod 15) for j=1..7
        """
        colors = [0] * num_edges
        for i in range(15):
            # Edge to center
            colors[edge_index[(15, i)]] = i
            for j in range(1, 8):
                a = (i + j) % 15
                b = (i - j) % 15
                if a > b:
                    a, b = b, a
                colors[edge_index[(a, b)]] = i
        return colors

    def get_random_colors(K):
        return [random.randint(0, K - 1) for _ in range(num_edges)]

    def calculate_initial_state(colors):
        """
        Calculates initial distinct color counts for each K4.
        Returns: k4_color_counts (list of dicts), k4_distinct (list of ints), current_energy
        """
        k4_color_counts = [{} for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        energy = 0
        
        for k4_id, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_id] = counts
            d = len(counts)
            k4_distinct[k4_id] = d
            if d < 5:
                energy += (5 - d) ** 2
        return k4_color_counts, k4_distinct, energy

    def simulate_annealing(init_colors, K, time_budget):
        colors = init_colors[:]
        k4_color_counts, k4_distinct, current_energy = calculate_initial_state(colors)
        
        best_energy = current_energy
        best_colors = colors[:]
        
        if current_energy == 0:
            return best_colors, best_energy

        T = 5.0
        cooling = 0.9995
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if current_energy == 0:
                break
                
            # Identify bad K4s
            bad_k4s = [i for i in range(num_k4s) if k4_distinct[i] < 5]
            if not bad_k4s:
                break
                
            # Strategy: Pick a random bad K4, then a random edge in it
            target_k4 = random.choice(bad_k4s)
            target_edge = random.choice(k4_edges[target_k4])
            
            old_color = colors[target_edge]
            # Try to pick a new color that is different from old
            # To speed up convergence, we can bias towards colors not present in the K4,
            # but random is safer for SA.
            new_color = random.randint(0, K - 1)
            
            if old_color == new_color:
                continue
                
            delta_energy = 0
            revert_data = []
            
            # Calculate delta energy
            for k4_id in edge_to_k4s[target_edge]:
                counts = k4_color_counts[k4_id]
                old_distinct = k4_distinct[k4_id]
                
                # Temporarily apply change to counts
                counts[old_color] -= 1
                if counts[old_color] == 0:
                    del counts[old_color]
                    
                if new_color in counts:
                    counts[new_color] += 1
                else:
                    counts[new_color] = 1
                    
                new_distinct = len(counts)
                
                old_pen = max(0, 5 - old_distinct) ** 2
                new_pen = max(0, 5 - new_distinct) ** 2
                delta_energy += new_pen - old_pen
                
                # Store state for potential revert
                # We need to restore counts and distinct
                # Instead of storing full dict, we store the changes
                # revert_data.append((k4_id, old_color, new_color, old_distinct, new_distinct, 
                #                     counts[old_color] if old_color in counts else 0,
                #                     counts[new_color] if new_color in counts else 0))
                
                # Simpler: Just record enough to revert
                # We modified counts in place.
                # To revert:
                # 1. counts[old_color] should be restored
                # 2. counts[new_color] should be restored
                # 3. k4_distinct should be restored
                
                # Let's just store the old values of the keys involved
                # But 'del' is tricky to revert without knowing if it was there.
                # Easier: Copy the affected entries? No, too slow.
                
                # Let's use a specific revert structure:
                # (k4_id, old_count_val, new_count_val, was_old_deleted, was_new_added)
                # This is getting complex. 
                # Alternative: Calculate delta, if reject, RECALCULATE distinct for affected K4s?
                # No, that's slow.
                
                # Let's stick to the in-place modification and careful revert.
                # State before:
                # old_count_old = counts.get(old_color, 0)
                # old_count_new = counts.get(new_color, 0)
                
                # We didn't store these. Let's restructure the loop to store them.
                # I will rewrite the inner loop to be safer.
                
            # The above logic is flawed because I modified 'counts' before deciding to revert.
            # Let's implement a cleaner revert mechanism.
            
            # RE-IMPLEMENTATION OF INNER LOOP FOR SAFETY
            
            # We'll break the loop and use a helper that returns delta and changes
            # Actually, for this specific problem, recalculating distinct for the ~15 affected K4s
            # on rejection is cheap enough if we do it correctly, or we store the old state.
            
            # Let's try a different approach:
            # 1. Calculate delta without modifying global state.
            # 2. If accepted, modify global state.
            # 3. If rejected, do nothing.
            
            # To calculate delta without modifying:
            # For each affected K4:
            #   old_d = k4_distinct[k4_id]
            #   old_counts = k4_color_counts[k4_id]
            #   new_counts = old_counts.copy() # Shallow copy of dict is fine for small dicts
            #   ... modify new_counts ...
            #   new_d = len(new_counts)
            #   delta += ...
            
            # This is O(K) per K4, K is small. 1820 K4s * 15 edges/K4? No, 1 edge affects ~15 K4s.
            # 15 * O(15) is cheap.
            
            delta_energy = 0
            changes = [] # Store (k4_id, old_color, new_color) for update
            
            for k4_id in edge_to_k4s[target_edge]:
                old_counts = k4_color_counts[k4_id]
                old_d = k4_distinct[k4_id]
                
                # Simulate change
                new_counts = old_counts.copy()
                
                # Remove old
                c_old = old_color
                if c_old in new_counts:
                    new_counts[c_old] -= 1
                    if new_counts[c_old] == 0:
                        del new_counts[c_old]
                        
                # Add new
                c_new = new_color
                if c_new in new_counts:
                    new_counts[c_new] += 1
                else:
                    new_counts[c_new] = 1
                    
                new_d = len(new_counts)
                
                old_pen = max(0, 5 - old_d) ** 2
                new_pen = max(0, 5 - new_d) ** 2
                delta_energy += new_pen - old_pen
                
                changes.append((k4_id, old_counts, new_counts, old_d, new_d))
            
            # Acceptance criterion
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                # Accept
                colors[target_edge] = new_color
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    
                # Commit changes
                for k4_id, old_counts, new_counts, old_d, new_d in changes:
                    k4_color_counts[k4_id] = new_counts
                    k4_distinct[k4_id] = new_d
            else:
                # Reject: No state change needed since we used copies for calculation
                pass
            
            T *= cooling
            
        return best_colors, best_energy

    def validate(colors):
        for k4_id, edges in enumerate(k4_edges):
            cols = set(colors[e] for e in edges)
            if len(cols) < 5:
                return False
        return True

    best_result = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Try 1-factorization (15 colors) with SA to fix violations.
    # 2. Try random 15 colors with SA.
    # 3. If all fail, try 16 colors? Or fallback to 17.
    
    # Time budget allocation
    total_time = 110.0
    time_per_run = 15.0
    num_runs = int(total_time // time_per_run)
    
    for run in range(num_runs):
        if time.time() > deadline - 5:
            break
            
        if run == 0:
            # Start with 1-factorization
            init = get_1factorization_colors()
            K = 15
        elif run == 1:
            # Perturbed 1-factorization
            init = get_1factorization_colors()
            # Randomly swap some colors
            for _ in range(10):
                e1 = random.randint(0, 119)
                e2 = random.randint(0, 119)
                init[e1], init[e2] = init[e2], init[e1]
            K = 15
        else:
            # Random start
            init = get_random_colors(15)
            K = 15
            
        colors, energy = simulate_annealing(init, K, time_per_run)
        
        if energy == 0:
            if validate(colors):
                # Count distinct colors used
                used = set(colors)
                num_used = len(used)
                if num_used < best_score:
                    best_score = num_used
                    best_result = colors[:]
                
                # If we found a valid 15-coloring, that's likely optimal (lower bound 13, but 15 is hard).
                # Can we try to reduce to 14 or 13?
                # The task says minimize. 
                # If we have a valid 15, we can try to compress it to 14/13 using a separate SA phase.
                
                if best_score <= 15:
                    # Try to compress
                    compression_budget = min(20, deadline - time.time() - 5)
                    if compression_budget > 5:
                        # Try to remap colors to 1..14 or 1..13
                        # This is a harder problem. 
                        # For now, returning 15 is a significant improvement over 17.
                        pass
                        
        # If we haven't found a valid 15, we might need to try K=16 or 17.
        # But 1-factorization is a strong start for 15.
        
    if best_result:
        return best_result
        
    # Fallback: 17-color modular solution
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

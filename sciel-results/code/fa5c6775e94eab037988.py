"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using bit-parallel validation + local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Fallback: known valid coloring for small n
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    def edge_index(i, j, n):
        """Convert (i,j) with i<j to linear index."""
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def build_adj(colors, n):
        """Build adjacency bitmasks for each color class."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj
    
    def count_c6s_bitset(adj, n):
        """Count monochromatic C6s using bit-parallel operations."""
        total = 0
        for c in range(4):
            ac = adj[c]
            for v0 in range(n):
                mask_v1 = ac[v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    if v1 <= v0:
                        continue
                    mask_v2 = ac[v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        if v2 <= v0:
                            continue
                        mask_v3 = ac[v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            if v3 <= v0:
                                continue
                            mask_v4 = ac[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                if v4 <= v0:
                                    continue
                                mask_v5 = ac[v4] & ac[v0]
                                used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used
                                if mask_v5:
                                    total += 1
        return total
    
    def is_valid(colors, n):
        """Check if coloring has no monochromatic C6."""
        adj = build_adj(colors, n)
        return count_c6s_bitset(adj, n) == 0
    
    def algebraic_coloring(n):
        """Deterministic algebraic construction using modular arithmetic."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Try several algebraic forms
                val = (i * j + 3 * i + 5 * j + 2) % 7
                c = val % 4
                colors.append(c)
        return colors
    
    def algebraic_coloring_v2(n):
        """Alternative algebraic construction."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                val = (diff * (diff + 1)) % 7
                c = val % 4
                colors.append(c)
        return colors
    
    def local_search_bitset(colors, n, time_limit):
        """Local search with bit-parallel C6 counting."""
        deadline_ls = time.time() + time_limit
        num_edges = n * (n - 1) // 2
        current_colors = colors[:]
        adj = build_adj(current_colors, n)
        current_count = count_c6s_bitset(adj, n)
        
        if current_count == 0:
            return current_colors, 0
        
        best_colors = current_colors[:]
        best_count = current_count
        rng = random.Random(42)
        
        # Precompute edge endpoints
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j))
        
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(50000):
            if time.time() > deadline_ls or time.time() > deadline:
                break
            if current_count == 0:
                return current_colors, 0
            if no_improve > max_no_improve:
                break
            
            # Select random edge
            ei = rng.randint(0, num_edges - 1)
            i, j = edge_list[ei]
            old_color = current_colors[ei]
            new_color = (old_color + 1 + rng.randint(0, 2)) % 4
            
            # Update adjacency
            adj[old_color][i] &= ~(1 << j)
            adj[old_color][j] &= ~(1 << i)
            adj[new_color][i] |= (1 << j)
            adj[new_color][j] |= (1 << i)
            current_colors[ei] = new_color
            
            new_count = count_c6s_bitset(adj, n)
            delta = new_count - current_count
            
            if delta <= 0:
                current_count = new_count
                no_improve = 0
                if new_count < best_count:
                    best_count = new_count
                    best_colors = current_colors[:]
            else:
                # Accept with some probability (simulated annealing)
                T = max(0.1, 3.0 - it * 0.0001)
                if rng.random() < np.exp(-delta / T):
                    current_count = new_count
                    no_improve = 0
                else:
                    # Revert
                    adj[old_color][i] |= (1 << j)
                    adj[old_color][j] |= (1 << i)
                    adj[new_color][i] &= ~(1 << j)
                    adj[new_color][j] &= ~(1 << i)
                    current_colors[ei] = old_color
                    no_improve += 1
        
        return best_colors, best_count
    
    # Try to find valid colorings for n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        
        num_edges = n * (n - 1) // 2
        found = False
        
        # Time allocation based on n
        remaining = deadline - time.time()
        if n >= 16:
            max_restarts = max(5, int(remaining * 2))
            time_per_restart = min(15.0, remaining / max(1, max_restarts))
        else:
            max_restarts = max(10, int(remaining * 3))
            time_per_restart = min(10.0, remaining / max(1, max_restarts))
        
        # Try algebraic initializations first
        init_candidates = [
            algebraic_coloring(n),
            algebraic_coloring_v2(n),
            [random.randint(0, 3) for _ in range(num_edges)],
        ]
        
        for init in init_candidates:
            if time.time() > deadline:
                break
            
            # Quick validation
            if is_valid(init, n):
                best_n = n
                best_colors = init
                found = True
                break
            
            # Local search from this initialization
            colors, count = local_search_bitset(init, n, time_per_restart)
            if count == 0:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if not found:
            # Try more random restarts
            for attempt in range(max_restarts - len(init_candidates)):
                if time.time() > deadline:
                    break
                
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                colors, count = local_search_bitset(colors, n, time_per_restart)
                if count == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break
        
        if found:
            continue  # Try to go higher
        else:
            # If failed at this n, we can try lower n but break if we're at 17
            if n == 17:
                continue
            else:
                break
    
    # Final validation of best result
    if best_n > 9:
        if not is_valid(best_colors, best_n):
            # Fallback to known valid coloring
            best_n = 9
            best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': best_n, 'colors': best_colors}


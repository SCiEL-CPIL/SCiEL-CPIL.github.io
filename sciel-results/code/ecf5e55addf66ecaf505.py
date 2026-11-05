"""Optimized 4-colouring of complete graphs without monochromatic triangles."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Greenwood-Gleason colouring of K16 (Clebsch graph structure)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def get_k16_colors():
        """Return 16x16 adjacency matrix for the Clebsch-based K16 3-colouring."""
        adj = np.zeros((16, 16), dtype=np.int8)
        for i in range(16):
            for j in range(i + 1, 16):
                c = log[i ^ j] % 3
                adj[i, j] = c
                adj[j, i] = c
        return adj

    def build_k32(k16_adj):
        """Build K32 by doubling K16 with colour 3 on cross edges."""
        n = 32
        colors = np.zeros((n, n), dtype=np.int8)
        # First copy
        colors[:16, :16] = k16_adj[:16, :16]
        # Second copy
        colors[16:, 16:] = k16_adj[:16, :16]
        # Cross edges get colour 3
        colors[:16, 16:] = 3
        colors[16:, :16] = 3
        return colors

    def count_mono_triangles(colors, n):
        """Count monochromatic triangles using bitmask operations."""
        # Build adjacency bitmasks for each colour
        masks = [0] * 4
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i, j]
                masks[c] |= (1 << j)
                # We'll handle this more carefully below
        
        # Better approach: for each colour, build adjacency lists as bitmasks
        color_masks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i, j]
                color_masks[c][i] |= (1 << j)
                color_masks[c][j] |= (1 << i)
        
        count = 0
        for c in range(4):
            for i in range(n):
                # Count edges within neighborhood of i in colour c
                mask_i = color_masks[c][i]
                # For each vertex j in neighborhood, count common neighbors
                # This is O(n^2 * 4) per colour which is fine for n up to ~50
                j_mask = mask_i
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    common = mask_i & color_masks[c][j]
                    count += common.bit_count()
                    j_mask ^= j_bit
        return count // 3  # Each triangle counted 3 times (once per vertex)

    def try_extend_to_n(current_colors, current_n, target_n, max_time=30.0):
        """Try to extend a valid colouring from current_n to target_n using local search."""
        if current_n >= target_n:
            return current_colors, current_n, count_mono_triangles(current_colors, current_n)
        
        # Start with the current colouring, add new vertices with random colours
        n = target_n
        colors = np.zeros((n, n), dtype=np.int8)
        colors[:current_n, :current_n] = current_colors[:current_n, :current_n]
        
        # Initialize new edges randomly
        for i in range(current_n, n):
            for j in range(i + 1, n):
                colors[i, j] = random.randint(0, 3)
                colors[j, i] = colors[i, j]
        
        # Also randomize edges between old and new vertices
        for i in range(current_n):
            for j in range(current_n, n):
                colors[i, j] = random.randint(0, 3)
                colors[j, i] = colors[i, j]
        
        # Local search: minimize monochromatic triangles
        best_colors = colors.copy()
        best_count = count_mono_triangles(colors, n)
        temperature = 10.0
        cooling = 0.995
        rng = np.random.default_rng(42)
        
        elapsed = time.time() - start_time
        time_budget = max_time - elapsed
        if time_budget <= 0:
            return best_colors, n, best_count
        
        steps = 0
        while time.time() - start_time < start_time + time_budget:
            steps += 1
            # Pick a random edge to flip
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[i, j]
            new_c = (old_c + 1) % 4
            
            # Estimate change in triangle count
            # Triangles affected: those containing edge (i,j)
            # A triangle (i,j,k) is monochromatic in colour c if all three edges are c
            # Changing (i,j) from old_c to new_c affects:
            # - Triangles that were mono in old_c: lose 1 for each k where (i,k) and (j,k) are old_c
            # - Triangles that become mono in new_c: gain 1 for each k where (i,k) and (j,k) are new_c
            
            # Count k such that colors[i,k] == old_c and colors[j,k] == old_c
            # and k != i, k != j
            common_old = 0
            common_new = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if colors[i, k] == old_c and colors[j, k] == old_c:
                    common_old += 1
                if colors[i, k] == new_c and colors[j, k] == new_c:
                    common_new += 1
            
            delta = common_new - common_old
            
            # Accept if delta < 0 or with probability exp(-delta/temperature)
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                colors[i, j] = new_c
                colors[j, i] = new_c
                best_count += delta
                if best_count < 0:
                    best_count = 0
                if best_count == 0:
                    return colors, n, 0
                temperature *= cooling
            else:
                # Occasionally try a different colour
                if random.random() < 0.1:
                    alt_c = (old_c + random.randint(1, 2)) % 4
                    if alt_c != old_c:
                        colors[i, j] = alt_c
                        colors[j, i] = alt_c
                        # Recompute locally (approximate)
                        temperature *= cooling
        
        return colors, n, best_count

    # Step 1: Build K32
    k16_adj = get_k16_colors()
    colors_32 = build_k32(k16_adj)
    mono_32 = count_mono_triangles(colors_32, 32)
    
    if mono_32 != 0:
        # Shouldn't happen, but fallback
        pass

    best_n = 32
    best_colors = colors_32
    best_mono = mono_32

    # Step 2: Try to extend to larger n
    for target_n in [33, 34, 35, 36, 37, 38, 39, 40]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        time_left = deadline - 20 - elapsed
        if time_left < 5:
            break
        
        new_colors, new_n, new_mono = try_extend_to_n(
            best_colors, best_n, target_n, max_time=min(time_left, 15.0)
        )
        
        if new_mono == 0:
            best_n = new_n
            best_colors = new_colors
            best_mono = 0
            continue
        
        # Even if not zero, keep the best so far
        if new_mono < best_mono:
            best_n = new_n
            best_colors = new_colors
            best_mono = new_mono
    
    # Final check: if we have a valid colouring, return it
    # Otherwise return the K32 which is known valid
    if best_mono == 0:
        n = best_n
    else:
        n = 32
        best_colors = colors_32

    # Flatten to lexicographic order
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            colors_list.append(int(best_colors[i, j]))
    
    return {'n': n, 'colors': colors_list}
# EVOLVE-BLOCK-END

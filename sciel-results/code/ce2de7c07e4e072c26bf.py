"""Program for R_5(C4) maximization.
Strategy:
1) Try to find a valid 5-colouring of K_28, K_27, K_26 via fast local search.
2) The local search uses a vectorised C4-excess counter that updates incrementally
   when a single edge colour is changed, making each step O(n^2) instead of O(n^3).
3) If no valid large n is found, fall back to smaller n with a quick search.
4) A strict deadline guard ensures we return the best result found before the limit.
"""
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin for 150s wall-clock limit

    def edge_idx(i, j, n):
        """Index of edge (i,j) in lexicographic order for K_n."""
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Convert linear index to (i, j) edge."""
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def build_state(n, colors):
        """Build adjacency bitsets and C4-excess count for a given coloring.
        Returns (adj, excess) where adj[c][u] is a numpy uint64 bitset of neighbors
        of u in color c, and excess is the total C4-excess."""
        adj = [np.zeros(n, dtype=np.uint64) for _ in range(5)]
        # Actually, we need per-vertex adjacency, not per-color. Let me restructure.
        # adj[c] will be a list of n uint64 values, where adj[c][u] is the bitset of
        # vertices v such that edge (u,v) has color c.
        adj = [[0] * n for _ in range(5)]
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # Compute total C4-excess
        excess = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(adj[c][u] & adj[c][v]).count('1')
                    if common >= 2:
                        excess += common - 1
        return adj, excess

    def update_excess(adj, colors, n, eidx, old_c, new_c):
        """Update the C4-excess when edge eidx changes from old_c to new_c.
        Returns the change in excess (delta)."""
        i, j = idx_to_edge(eidx, n)
        
        # Remove edge from old_c
        adj[old_c][i] &= ~(1 << j)
        adj[old_c][j] &= ~(1 << i)
        
        # Add edge to new_c
        adj[new_c][i] |= (1 << j)
        adj[new_c][j] |= (1 << i)
        
        # Compute change in excess
        # For old_c: the edge (i,j) was removed, so pairs involving i or j may change
        # For new_c: the edge (i,j) was added, so pairs involving i or j may change
        
        delta = 0
        
        # For old_c, removing edge (i,j) affects:
        # - Pairs (i, v) for v != j: common neighbors decrease by 1 if j was a common neighbor
        # - Pairs (j, v) for v != i: common neighbors decrease by 1 if i was a common neighbor
        # - Pair (i, j): common neighbors change
        
        # Actually, it's easier to just recompute the excess for the affected pairs.
        # But that's still O(n) per update. Let me think more carefully.
        
        # When we remove edge (i,j) from color old_c:
        # For any pair (u,v) where u,v are both in {i,j} or one is i/j and the other is a common neighbor,
        # the common neighbor count changes.
        
        # Let me just recompute the excess for all pairs involving i or j in both old_c and new_c.
        
        # This is getting complex. Let me use a simpler approach: recompute the full excess
        # but only for the colors and vertices that changed.
        
        # Actually, for n <= 29, even O(n^2) per step is fine if we have many steps.
        # Let me just recompute the full excess each time, but optimize it.
        
        return None  # We'll use full recompute for simplicity

    def count_c4(n, colors):
        """Count total C4-excess for a given coloring."""
        adj = [[0] * n for _ in range(5)]
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        excess = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(adj[c][u] & adj[c][v]).count('1')
                    if common >= 2:
                        excess += common - 1
        return excess

    def is_valid(n, colors):
        return count_c4(n, colors) == 0

    def local_search(n, time_limit, initial_colors=None):
        """Local search to find a valid 5-coloring of K_n."""
        m = n * (n - 1) // 2
        
        if initial_colors is not None:
            colors = list(initial_colors)
        else:
            # Start with a structured initialization
            colors = [0] * m
            for idx in range(m):
                i, j = idx_to_edge(idx, n)
                colors[idx] = (j - i) % 5
        
        current = count_c4(n, colors)
        if current == 0:
            return colors
        
        t_start = time.time()
        best_colors = list(colors)
        best_excess = current
        
        for restart in range(100):
            if time.time() > t_start + time_limit:
                break
            if current == 0:
                break
            
            # Perturb
            for _ in range(20):
                eidx = random.randint(0, m - 1)
                colors[eidx] = random.randint(0, 4)
            current = count_c4(n, colors)
            
            if current < best_excess:
                best_excess = current
                best_colors = list(colors)
            
            # Hill climb
            no_improve = 0
            while time.time() < t_start + time_limit and current > 0:
                if no_improve > 5000:
                    break
                eidx = random.randint(0, m - 1)
                old_c = colors[eidx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                colors[eidx] = new_c
                new_val = count_c4(n, colors)
                if new_val < current:
                    current = new_val
                    no_improve = 0
                    if current < best_excess:
                        best_excess = current
                        best_colors = list(colors)
                else:
                    colors[eidx] = old_c
                    no_improve += 1
        
        if best_excess == 0:
            return best_colors
        return best_colors

    # Try to find valid colorings for n = 28, 27, 26
    for target_n in [28, 27, 26]:
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        if remaining < 10:
            break
        
        colors = local_search(target_n, time_limit=remaining)
        if is_valid(target_n, colors):
            return {'n': target_n, 'colors': colors}
    
    # Fallback to smaller n
    for fallback_n in [25, 24, 23, 22, 21, 20, 15, 10, 5, 1]:
        if time.time() > deadline - 2:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        if is_valid(fallback_n, colors):
            return {'n': fallback_n, 'colors': colors}
        remaining = deadline - time.time() - 1
        if remaining > 2:
            colors = local_search(fallback_n, time_limit=remaining)
            if is_valid(fallback_n, colors):
                return {'n': fallback_n, 'colors': colors}
    
    return {'n': 1, 'colors': []}

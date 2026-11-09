import time
import random
import numpy as np

def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    # ---------- Edge indexing helpers ----------

    def edge_idx(i, j, n):
        """Lexicographic index of edge (i,j) with i<j in a list of length n*(n-1)//2."""
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Reverse: given linear index idx (0-based), return (i,j) with i<j."""
        # Use binary search or precomputed offsets for speed if n is large
        # For n<=30, linear scan is fine, but let's optimize slightly
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    # ---------- Incremental C4 Counter ----------
    
    class C4Counter:
        def __init__(self, n):
            self.n = n
            self.m = n * (n - 1) // 2
            self.colors = [-1] * self.m
            # neighbors[c][u] = list of neighbors of u in color c
            self.neighbors = [[[] for _ in range(n)] for _ in range(5)]
            # common[c][u*n + v] for u < v: number of common neighbors in color c
            # We use a flat array for speed
            self.common = [np.zeros(n * n, dtype=np.int32) for _ in range(5)]
            self.total = 0  # total excess C4 count (sum of max(0, common-1))

        def _common_idx(self, u, v):
            if u > v:
                u, v = v, u
            return u * self.n + v

        def _add_common(self, c, u, v, delta):
            idx = self._common_idx(u, v)
            old = self.common[c][idx]
            new = old + delta
            self.common[c][idx] = new
            old_excess = old - 1 if old > 1 else 0
            new_excess = new - 1 if new > 1 else 0
            self.total += new_excess - old_excess

        def assign(self, eidx, c):
            """Assign color c to edge eidx."""
            n = self.n
            i, j = idx_to_edge(eidx, n)
            old_c = self.colors[eidx]
            if old_c == c:
                return
            if old_c != -1:
                self._remove_edge(i, j, old_c)
            self._add_edge(i, j, c)
            self.colors[eidx] = c

        def _add_edge(self, u, v, c):
            n = self.n
            # Update common counts for all pairs involving u or v
            # Adding edge (u,v) in color c:
            # For each w in N_c(u) (w != v): common[c][v][w] += 1
            # For each w in N_c(v) (w != u): common[c][u][w] += 1
            
            nu = self.neighbors[c][u]
            nv = self.neighbors[c][v]
            
            for w in nu:
                if w != v:
                    self._add_common(c, v, w, 1)
            for w in nv:
                if w != u:
                    self._add_common(c, u, w, 1)
            
            # Update neighbor lists
            self.neighbors[c][u].append(v)
            self.neighbors[c][v].append(u)

        def _remove_edge(self, u, v, c):
            n = self.n
            # Reverse of _add_edge
            nu = self.neighbors[c][u]
            nv = self.neighbors[c][v]
            
            for w in nu:
                if w != v:
                    self._add_common(c, v, w, -1)
            for w in nv:
                if w != u:
                    self._add_common(c, u, w, -1)
            
            # Remove from neighbor lists
            # List removal is O(n), but n is small
            if v in self.neighbors[c][u]:
                self.neighbors[c][u].remove(v)
            if u in self.neighbors[c][v]:
                self.neighbors[c][v].remove(u)

    # ---------- Local Search ----------

    def local_search(n, time_limit, seed=42):
        """Local search: minimise C4 count. Returns (colors, valid_flag)."""
        random.seed(seed)
        m = n * (n - 1) // 2
        
        # Initialize randomly
        colors = [random.randint(0, 4) for _ in range(m)]
        
        counter = C4Counter(n)
        for idx in range(m):
            counter.assign(idx, colors[idx])
            
        current_total = counter.total
        if current_total == 0:
            return colors, True

        t_start = time.time()
        no_improve_count = 0
        max_no_improve = 5000 # Reset if stuck? Or just keep going.
        
        while time.time() < t_start + time_limit:
            if current_total == 0:
                return colors, True
            
            # Pick a random edge to change
            eidx = random.randint(0, m - 1)
            old_c = colors[eidx]
            
            # Try all 5 colors, pick the best
            best_delta = float('inf')
            best_c = -1
            
            i, j = idx_to_edge(eidx, n)
            
            for c in range(5):
                if c == old_c:
                    continue
                
                # Simulate change
                # We need to calculate the change in total without actually modifying state
                # This is complex. Instead, we can just do a "walk":
                # Try one random new color. If it improves, accept. If not, reject.
                # This is simpler and often sufficient with many iterations.
                pass
            
            # Simplified Walk: Pick a random new color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta efficiently?
            # The C4Counter doesn't support "dry run".
            # We must apply, check, and revert if worse.
            
            counter.assign(eidx, new_c)
            new_total = counter.total
            
            if new_total < current_total:
                current_total = new_total
                colors[eidx] = new_c
                no_improve_count = 0
            else:
                # Revert
                counter.assign(eidx, old_c)
                colors[eidx] = old_c
                no_improve_count += 1
                
                # If stuck for too long, try a random restart or perturbation?
                # For now, just keep going.
                
        return colors, (current_total == 0)

    # ---------- Main Logic ----------

    # Try n from 27 down to 1
    # We prioritize higher n.
    # Lazebnik-Woldar suggests 26 is possible. 27 is unknown but possible?
    # Let's try 27, 26, 25.
    
    best_n = 0
    best_colors = None
    
    # Time allocation:
    # n=27: 50s
    # n=26: 50s
    # n=25: 30s
    # n=24: 10s
    # Rest: 5s
    
    for n in range(27, 0, -1):
        if time.time() > deadline:
            break
            
        # Determine time budget for this n
        if n >= 27:
            budget = 50.0
        elif n >= 26:
            budget = 50.0
        elif n >= 25:
            budget = 30.0
        elif n >= 24:
            budget = 10.0
        else:
            budget = 5.0
            
        # Ensure we don't exceed deadline
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        budget = min(budget, remaining - 0.5)
        
        # Run local search
        # Use multiple restarts if time permits
        colors, valid = local_search(n, budget, seed=n)
        
        if valid:
            return {'n': n, 'colors': colors}
        
        # If not valid, store as best candidate if n > best_n
        if n > best_n:
            best_n = n
            best_colors = colors.copy()
            
    # If no valid coloring found, return the best attempt (invalid) or a known small valid one?
    # The task says "Maximize n". An invalid result scores 0.
    # We MUST return a valid result.
    # So we need to fallback to a smaller n that is likely valid.
    # n=1 is always valid.
    # n=2 is always valid.
    # n=3 is always valid.
    # n=4: K4 has 6 edges. 5 colors. Can we avoid C4?
    # K4 has no C4 (needs 4 vertices, but C4 is a cycle of length 4. K4 has triangles and 4-cycles?
    # Vertices 0,1,2,3. Cycle 0-1-2-3-0. Edges (0,1), (1,2), (2,3), (3,0).
    # If all 4 edges are same color, it's a C4.
    # With 5 colors, we can easily color K4 without monochromatic C4.
    # So n=4 is valid.
    
    # Let's find a small valid n quickly to be safe.
    # We can just return n=4 with a simple coloring.
    # Or if best_n >= 4 and best_colors is not None, we can't use it if invalid.
    # We need a valid coloring.
    
    # Quick check for small n
    for n in range(4, 0, -1):
        if time.time() > deadline:
             break
        # Just try to find a valid one for small n
        # For n<=4, random search is very fast
        for _ in range(1000):
            colors, valid = local_search(n, 0.1, seed=random.randint(0, 1000))
            if valid:
                return {'n': n, 'colors': colors}
                
    # Fallback: n=1
    return {'n': 1, 'colors': []}

# Test run
if __name__ == "__main__":
    res = construct()
    print(res['n'])

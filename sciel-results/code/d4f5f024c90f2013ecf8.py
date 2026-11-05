import time
import random
import sys

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    
    # Precompute edge index mapping for fast access
    # edge_idx(i, j) for i < j in lexicographic order
    # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(2*n-1-i)//2
    # Within row i, edge to j (j > i) is at offset (j - i - 1)
    
    def make_edge_indexer(n):
        """Return a function edge_idx(i, j) -> index for i < j"""
        # Precompute row offsets
        row_offset = [0] * (n + 1)
        for i in range(n):
            row_offset[i + 1] = row_offset[i] + (n - 1 - i)
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return row_offset[i] + (j - i - 1)
        return edge_idx, row_offset
    
    def count_violations_fast(n, colors):
        """Count C4 violations efficiently using bitsets."""
        if n > 64:
            # Use list of sets for large n
            total = 0
            for c in range(5):
                adj = [set() for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == c:
                            adj[i].add(j)
                            adj[j].add(i)
                        idx += 1
                for i in range(n):
                    for j in range(i + 1, n):
                        if len(adj[i] & adj[j]) >= 2:
                            total += 1
            return total
        else:
            # Use Python integers as bitsets (arbitrary precision)
            total = 0
            for c in range(5):
                adj = [0] * n  # adj[i] is a bitset of neighbors in color c
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == c:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                        idx += 1
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        # Count bits in common
                        if common & (common - 1):  # at least 2 bits set
                            total += 1
            return total
    
    def local_search_bitset(n, colors, max_iter, seed, deadline):
        """Local search using bitset-based violation counting."""
        rng = random.Random(seed)
        colors = list(colors)
        m = n * (n - 1) // 2
        best = colors[:]
        best_viol = count_violations_fast(n, best)
        
        # Precompute row offsets for edge indexing
        row_offset = [0] * (n + 1)
        for i in range(n):
            row_offset[i + 1] = row_offset[i] + (n - 1 - i)
        
        # Precompute vertex pairs for each edge index
        edge_vertices = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_vertices.append((i, j))
                idx += 1
        
        for it in range(max_iter):
            if best_viol == 0:
                break
            if time.time() > deadline - 5:
                break
            
            idx = rng.randint(0, m - 1)
            old = colors[idx]
            new = (old + rng.randint(1, 4)) % 5
            colors[idx] = new
            v = count_violations_fast(n, colors)
            if v <= best_viol:
                best = colors[:]
                best_viol = v
            else:
                colors[idx] = old
        
        return best, best_viol
    
    def simulated_annealing(n, colors, max_iter, seed, deadline):
        """Simulated annealing for C4-free coloring."""
        rng = random.Random(seed)
        colors = list(colors)
        m = n * (n - 1) // 2
        best = colors[:]
        best_viol = count_violations_fast(n, best)
        
        edge_vertices = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_vertices.append((i, j))
        
        T = 1.0
        cooling = 0.9995
        
        for it in range(max_iter):
            if best_viol == 0:
                break
            if time.time() > deadline - 5:
                break
            
            idx = rng.randint(0, m - 1)
            old = colors[idx]
            new = (old + rng.randint(1, 4)) % 5
            colors[idx] = new
            v = count_violations_fast(n, colors)
            
            if v <= best_viol:
                best = colors[:]
                best_viol = v
            elif rng.random() < (1.0 if v - best_viol <= 1 else 0.1):
                # Accept worse solutions with some probability
                pass
            else:
                colors[idx] = old
            
            T *= cooling
        
        return best, best_viol
    
    # Start with a small known-good coloring and grow
    # For n=5, any coloring works (no C4 possible in K5 with 5 colors if we're careful)
    # Actually for n <= 5, trivially no C4
    
    # Use a greedy construction to get a valid small coloring
    def greedy_color(n, seed):
        """Greedy edge coloring avoiding C4."""
        rng = random.Random(seed)
        colors = [0] * (n * (n - 1) // 2)
        edge_vertices = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_vertices.append((i, j))
        
        # Build adjacency incrementally
        adj = [set() for _ in range(5)]
        for idx, (i, j) in enumerate(edge_vertices):
            # Try each color
            valid = []
            for c in range(5):
                # Check if adding edge (i,j) in color c creates a C4
                ok = True
                for x in adj[c][i]:
                    if x in adj[c][j]:
                        # i and j share common neighbor x in color c
                        # Check if there's another common neighbor
                        common = adj[c][i] & adj[c][j]
                        if len(common) >= 1:
                            ok = False
                            break
                if ok:
                    valid.append(c)
            
            if valid:
                c = rng.choice(valid)
            else:
                c = rng.randint(0, 4)
            
            colors[idx] = c
            adj[c][i].add(j)
            adj[c][j].add(i)
        
        return colors
    
    # Best known: try to build up from small n
    best_n = 5
    best_colors = greedy_color(5, 42)
    
    # Verify initial
    if count_violations_fast(5, best_colors) != 0:
        # Fallback: just use a simple valid coloring
        best_colors = [0] * 10
        best_n = 5
    
    # Try to grow
    for target_n in range(6, 27):
        if time.time() > deadline - 20:
            break
        
        found = False
        remaining = deadline - time.time()
        
        # Multiple strategies
        strategies = [
            ('greedy', lambda: greedy_color(target_n, attempt * 1000 + target_n)),
            ('random_ls', None),
            ('sa', None),
        ]
        
        num_attempts = max(3, int(remaining / 20))
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            if attempt % 3 == 0:
                # Greedy
                colors = greedy_color(target_n, attempt * 104729 + target_n * 7919)
                colors, viol = local_search_bitset(target_n, colors, max_iter=500, seed=attempt * 31, deadline=deadline)
            elif attempt % 3 == 1:
                # Random + local search
                rng = random.Random(attempt * 104729 + target_n * 7919)
                colors = [rng.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
                colors, viol = local_search_bitset(target_n, colors, max_iter=500, seed=attempt * 31, deadline=deadline)
            else:
                # Greedy + simulated annealing
                colors = greedy_color(target_n, attempt * 104729 + target_n * 7919)
                colors, viol = simulated_annealing(target_n, colors, max_iter=500, seed=attempt * 31, deadline=deadline)
            
            if viol == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if not found and target_n >= 20:
            # If we can't find valid colorings for large n, stop
            # But try a bit more
            if target_n >= 24:
                break
    
    return {'n': best_n, 'colors': best_colors}

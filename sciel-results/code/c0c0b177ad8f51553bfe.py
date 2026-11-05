import time
import random

def construct():
    deadline = time.time() + 110
    
    class C4Counter:
        def __init__(self, n, colors):
            self.n = n
            self.colors = list(colors)
            self.adj = [set() for _ in range(5)]  # adj[c][i] = set of neighbors of i in color c
            self.common = [[0]*n for _ in range(n)]  # common[i][j] = total violations for pair (i,j)
            self.pair_common = [[[0]*n for _ in range(n)] for _ in range(5)]  # pair_common[c][i][j]
            
            # Build adjacency
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = self.colors[idx]
                    self.adj[c][i].add(j)
                    self.adj[c][j].add(i)
                    idx += 1
            
            # Compute common neighbors
            self.total_violations = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        cn = len(self.adj[c][i] & self.adj[c][j])
                        self.pair_common[c][i][j] = cn
                        self.pair_common[c][j][i] = cn
                        if cn >= 2:
                            self.total_violations += 1
                            self.common[i][j] += 1
                            self.common[j][i] += 1
        
        def change_edge(self, edge_idx, new_color):
            """Change the color of edge edge_idx to new_color. Returns change in total violations."""
            n = self.n
            # Find which vertices this edge connects
            # edge_idx corresponds to (i,j) in lexicographic order
            # i*(2n-1-i)//2 + (j-i-1) = edge_idx
            # This is complex; let me precompute
            i, j = self.edge_vertices[edge_idx]
            old_color = self.colors[edge_idx]
            if old_color == new_color:
                return 0
            
            # Remove edge from old_color
            self.adj[old_color][i].discard(j)
            self.adj[old_color][j].discard(i)
            
            # Update pair_common for old_color: pairs involving i or j
            delta = 0
            for x in range(n):
                if x == i or x == j:
                    continue
                # Pair (i, x): common neighbors changed because j is no longer a neighbor of i
                # Actually, we need to recompute common neighbors for pairs (i,x) and (j,x)
                # This is getting complicated. Let me just recompute affected pairs.
            
            # This is getting too complex. Let me just recompute the full count.
            self.adj[old_color][i].add(j)
            self.adj[old_color][j].add(i)
            
            return 0  # Not implementing incremental for now
    
    # This incremental approach is too complex. Let me just use the simple approach
    # but optimize the C4 count.
    
    def count_violations(n, colors):
        total = 0
        for c in range(5):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            for i in range(n):
                for j in range(i+1, n):
                    if len(adj[i] & adj[j]) >= 2:
                        total += 1
        return total
    
    def local_search(n, colors, max_iter=2000, seed=None):
        rng = random.Random(seed)
        colors = list(colors)
        m = n*(n-1)//2
        best = colors[:]
        best_viol = count_violations(n, best)
        
        for it in range(max_iter):
            if best_viol == 0:
                break
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, m - 1)
            old = colors[idx]
            new = (old + rng.randint(1, 4)) % 5
            colors[idx] = new
            v = count_violations(n, colors)
            if v <= best_viol:
                best = colors[:]
                best_viol = v
            else:
                colors[idx] = old
        
        return best, best_viol
    
    n0 = 11
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = colors0
    
    for target_n in range(12, 27):
        if time.time() > deadline - 15:
            break
        
        found = False
        remaining = deadline - time.time()
        num_attempts = max(2, int(remaining / 15))
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            rng = random.Random(attempt * 104729 + target_n * 7919)
            colors = [rng.randint(0, 4) for _ in range(target_n*(target_n-1)//2)]
            
            colors, viol = local_search(target_n, colors, max_iter=1000, seed=attempt*31)
            
            if viol == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if not found and target_n >= 20:
            break
    
    return {'n': best_n, 'colors': best_colors}

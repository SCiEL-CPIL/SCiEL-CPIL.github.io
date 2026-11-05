import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 18
    
    def make_k666():
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if i // 6 != j // 6:
                    adj[i][j] = adj[j][i] = True
        return adj
    
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free_size(adj):
        # Use branch and bound
        best = 0
        # Try to find if there's a triangle-free set of size k
        for k in range(12, 0, -1):
            if _has_tf_set(adj, k):
                return k
        return 0
    
    def _has_tf_set(adj, k):
        # Backtracking: build a set with no triangles
        def backtrack(candidates, size):
            if size == 0:
                return True
            if len(candidates) < size:
                return False
            if size == 1:
                return True
            # Pick a vertex, either include or exclude
            # Heuristic: try vertices with fewer neighbors in candidates
            v = candidates[0]
            # Include v: new candidates must not form triangle with v
            # i.e., for any two remaining vertices, they can't both be adjacent to v
            # Actually, we need: no triangle in the final set
            # If we include v, then for any pair (x,y) in the rest, not (adj[v][x] and adj[v][y] and adj[x][y])
            # Simpler: the remaining set must be triangle-free AND contain no edge between two neighbors of v
            # Even simpler: just recurse with filtered candidates
            new_cands = []
            for u in candidates[1:]:
                new_cands.append(u)
            # Filter: remove vertices that would form triangle with v and another chosen vertex
            # This is complex; let's just do simple backtracking
            if backtrack(new_cands, size - 1):
                return True
            # Exclude v
            if backtrack(candidates[1:], size):
                return True
            return False
        return backtrack(list(range(n)), k)
    
    def get_edges(adj):
        return [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    adj = make_k666()
    best_adj = [row[:] for row in adj]
    best_score = 12  # known
    
    random.seed(42)
    t = 50
    cooling = 0.9995
    
    current_adj = [row[:] for row in adj]
    current_score = best_score
    
    while time.time() < deadline and t > 1:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        add = not current_adj[i][j]
        old = current_adj[i][j]
        current_adj[i][j] = current_adj[j][i] = add
        
        if not is_k4_free(current_adj):
            current_adj[i][j] = current_adj[j][i] = old
            t *= cooling
            continue
        
        new_score = max_triangle_free_size(current_adj)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / t):
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in current_adj]
            current_score = new_score
        else:
            current_adj[i][j] = current_adj[j][i] = old
        
        t *= cooling
    
    return get_edges(best_adj)

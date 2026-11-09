import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute color pairs
    # We will try k = 3, 4, 5
    
    def try_coloring(k, seed=0):
        random.seed(seed)
        colors = [-1] * n
        # UnionFind for each pair of colors
        # There are k*(k-1)/2 pairs.
        # Index pairs by (min(c1,c2), max(c1,c2))
        # We need to map pair to a UnionFind instance.
        # Since n=50, we can just have a dict.
        ufs = {}
        
        # Order of vertices to color. 
        # Maybe random order? Or degree order? 
        # Let's try random order for restarts.
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            # Try each color in random order
            candidate_colors = list(range(k))
            random.shuffle(candidate_colors)
            
            valid_color = -1
            for c in candidate_colors:
                # Check if c is valid for v
                is_valid = True
                # Store unions to undo if we fail later? 
                # No, we decide one vertex at a time. 
                # But we need to check against ALL neighbors.
                # If we pick c, we must union v with all neighbors u having color c'.
                # If any union fails (cycle), c is invalid.
                
                # First, check if any neighbor has color c
                for u in adj[v]:
                    if colors[u] == c:
                        is_valid = False
                        break
                if not is_valid:
                    continue
                    
                # Check for bicolored cycles
                # For each neighbor u with color c' != c
                # We need to see if v and u are already connected in UFs[c, c']
                # If they are, adding edge (v,u) creates a cycle.
                
                # We also need to check if adding v to the forest of (c, c') 
                # causes issues with other neighbors of v that have color c'?
                # Actually, the edge is (v,u). 
                # If find(v) == find(u) in UFS[c,c'], then cycle.
                # But v is not yet in the UF. 
                # We can simulate: 
                # If we add v, it becomes a new node in the UF (or we can just check connectivity among neighbors).
                # Wait, the subgraph induced by colors c and c' includes v and all neighbors u with color c'.
                # The edges are (v,u) for all such u.
                # If there are two neighbors u1, u2 with color c' such that u1 and u2 are already connected 
                # in the existing forest (excluding v), then adding v connects to both, creating a cycle v-u1-...-u2-v.
                # Also, if u1 and u2 are the same node, that's a multi-edge, but graph is simple.
                
                # So, for a fixed c, let N_c' be the set of neighbors u of v with color c'.
                # We need to check if any two vertices in N_c' are connected in UFS[c, c'].
                # If yes, cycle.
                
                # Also, we need to check if v itself creates a cycle with itself? No.
                
                # Let's optimize:
                # For each color c' present in neighbors of v:
                #   Get the set of neighbors with that color.
                #   Check if they are all in distinct components of UFS[c, c'].
                #   If any two are in the same component, invalid.
                
                # Additionally, we must ensure that the new edges (v,u) don't form a cycle 
                # involving paths that go through other colors? 
                # No, the bicolored subgraph only contains vertices of color c or c'.
                # So paths in UFS[c, c'] only use vertices of color c or c'.
                # Since v is new to color c, it wasn't in UFS[c, c'] before.
                # So we just need to check connectivity among the neighbors of color c'.
                
                # If valid so far, we tentatively accept c.
                # We need to actually perform the unions to see if it works?
                # If we accept c, we must update UFS.
                # But if we later find a problem, we can't easily undo.
                # However, we are processing vertices one by one. 
                # If we pick a color, we commit to it.
                # So we just need to verify that picking c is consistent with the *current* state.
                
                # Verification:
                # For each neighbor u with color c' != c:
                #   Let comp_u = UFS[c, c'].find(u)
                #   We need to ensure that no other neighbor u' of v with color c' 
                #   has comp_u' == comp_u.
                
                # Let's group neighbors by color.
                neigh_colors = {}
                for u in adj[v]:
                    cu = colors[u]
                    if cu == -1: continue
                    if cu == c:
                        is_valid = False
                        break
                    if cu not in neigh_colors:
                        neigh_colors[cu] = []
                    neigh_colors[cu].append(u)
                
                if not is_valid:
                    continue
                    
                for c_prime, us in neigh_colors.items():
                    if len(us) == 0: continue
                    # Get the pair key
                    c1, c2 = (c, c_prime) if c < c_prime else (c_prime, c)
                    key = (c1, c2)
                    if key not in ufs:
                        # No existing edges for this pair, so no cycles possible among neighbors
                        # (since they are isolated in this subgraph so far? 
                        # Wait, if there are no edges, they are all separate components.
                        # So no two can be connected.)
                        continue
                        
                    uf = ufs[key]
                    seen_roots = set()
                    for u in us:
                        root = uf.find(u)
                        if root in seen_roots:
                            is_valid = False
                            break
                        seen_roots.add(root)
                    if not is_valid:
                        break
                        
                if is_valid:
                    valid_color = c
                    break
            
            if valid_color == -1:
                return None # Failed to color vertex v
            
            colors[v] = valid_color
            
            # Update UFs
            for u in adj[v]:
                cu = colors[u]
                if cu == -1: continue
                if cu == valid_color: continue # Should not happen
                
                c1, c2 = (valid_color, cu) if valid_color < cu else (cu, valid_color)
                key = (c1, c2)
                if key not in ufs:
                    ufs[key] = UnionFind(n)
                    # Initialize? 
                    # The UF is over all n vertices, but only those with color c1 or c2 are "active".
                    # We don't need to initialize all. find/union will work.
                
                uf = ufs[key]
                # Union v and u
                # Note: If this returns False, it means a cycle was formed.
                # But we already checked that no two neighbors of v with color cu are connected.
                # Does that guarantee that union(v,u) won't fail?
                # Yes, because v is a new node in this UF (it hasn't been added to this UF before).
                # So find(v) is v. find(u) is some root.
                # They are different. So union will succeed.
                # Wait, is v "new"? 
                # v is being colored valid_color for the first time.
                # So it has no edges in any bicolored subgraph yet.
                # So yes, it is a new node in all UFs involving valid_color.
                
                uf.union(v, u)
                
        return colors

    best_colors = None
    best_k = 50
    
    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            res = try_coloring(k, seed=attempt)
            if res is not None:
                best_colors = res
                best_k = k
                found = True
                break # Found a valid coloring for this k, move to next k? 
                # Actually, we want the MINIMUM k. 
                # So if we find one for k=3, we should return it.
                # If not, try k=4.
                # If not, try k=5.
        
        if found:
            # We found a valid coloring with k colors.
            # Since we try in increasing order, this is the minimum k found.
            # We can return immediately.
            break
            
    if best_colors is None:
        # Fallback: use 50 colors (always valid)
        best_colors = list(range(n))
        
    # Ensure types are correct
    return [int(c) for c in best_colors]

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 0
    best_colors = []
    
    # Precompute triples for each n up to 40
    triples_by_n = {}
    for n in range(10, 41):
        triples_by_n[n] = list(combinations(range(n), 3))
    
    def build_adj(n, colors):
        """Build red_adj and blue_adj from colors."""
        red_adj = [[set() for _ in range(n)] for __ in range(n)]
        blue_adj = [[set() for _ in range(n)] for __ in range(n)]
        triples = triples_by_n[n]
        for i, (a, b, c) in enumerate(triples):
            if colors[i] == 0:  # Red
                red_adj[a][b].add(c); red_adj[b][a].add(c)
                red_adj[a][c].add(b); red_adj[c][a].add(b)
                red_adj[b][c].add(a); red_adj[c][b].add(a)
            else:  # Blue
                blue_adj[a][b].add(c); blue_adj[b][a].add(c)
                blue_adj[a][c].add(b); blue_adj[c][a].add(b)
                blue_adj[b][c].add(a); blue_adj[c][b].add(a)
        return red_adj, blue_adj
    
    def find_red_k4(n, red_adj):
        """Find a red K4. Returns (u,v,w1,w2) or None."""
        for u in range(n):
            for v in range(u+1, n):
                if len(red_adj[u][v]) >= 2:
                    w_list = list(red_adj[u][v])
                    for i in range(len(w_list)):
                        for j in range(i+1, len(w_list)):
                            w1, w2 = w_list[i], w_list[j]
                            if w2 in red_adj[u][w1] and w1 in red_adj[v][w2]:
                                return (u, v, w1, w2)
        return None
    
    def find_blue_k5(n, blue_adj):
        """Find a blue K5. Returns a 5-tuple or None."""
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            # Check all 10 triples
            if (c in blue_adj[a][b] and b in blue_adj[a][c] and a in blue_adj[b][c] and
                d in blue_adj[a][b] and b in blue_adj[a][d] and a in blue_adj[b][d] and
                c in blue_adj[a][d] and d in blue_adj[a][c] and a in blue_adj[c][d] and
                b in blue_adj[c][d] and d in blue_adj[b][c] and c in blue_adj[b][d] and
                e in blue_adj[a][b] and b in blue_adj[a][e] and a in blue_adj[b][e] and
                c in blue_adj[a][e] and e in blue_adj[a][c] and a in blue_adj[c][e] and
                d in blue_adj[a][e] and e in blue_adj[a][d] and a in blue_adj[d][e] and
                b in blue_adj[c][e] and e in blue_adj[b][c] and c in blue_adj[b][e] and
                d in blue_adj[c][e] and e in blue_adj[c][d] and c in blue_adj[d][e]):
                return combo
        return None
    
    def flip_triple(n, colors, red_adj, blue_adj, triple):
        """Flip the color of a triple and update adjacency structures."""
        a, b, c = triple
        # Find index
        triples = triples_by_n[n]
        # Linear search - could be optimized with dict
        idx = None
        for i, t in enumerate(triples):
            if t == triple:
                idx = i
                break
        if idx is None:
            return False
        
        old_color = colors[idx]
        new_color = 1 - old_color
        colors[idx] = new_color
        
        # Remove from old adjacency
        if old_color == 0:  # Was red
            red_adj[a][b].discard(c); red_adj[b][a].discard(c)
            red_adj[a][c].discard(b); red_adj[c][a].discard(b)
            red_adj[b][c].discard(a); red_adj[c][b].discard(a)
        else:  # Was blue
            blue_adj[a][b].discard(c); blue_adj[b][a].discard(c)
            blue_adj[a][c].discard(b); blue_adj[c][a].discard(b)
            blue_adj[b][c].discard(a); blue_adj[c][b].discard(a)
        
        # Add to new adjacency
        if new_color == 0:  # Now red
            red_adj[a][b].add(c); red_adj[b][a].add(c)
            red_adj[a][c].add(b); red_adj[c][a].add(b)
            red_adj[b][c].add(a); red_adj[c][b].add(a)
        else:  # Now blue
            blue_adj[a][b].add(c); blue_adj[b][a].add(c)
            blue_adj[a][c].add(b); blue_adj[c][a].add(b)
            blue_adj[b][c].add(a); blue_adj[c][b].add(a)
        
        return True
    
    def is_valid(n, colors, red_adj, blue_adj):
        """Check if coloring is valid (no red K4, no blue K5)."""
        if find_red_k4(n, red_adj) is not None:
            return False
        if find_blue_k5(n, blue_adj) is not None:
            return False
        return True
    
    # Try to find valid colorings for increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        found = False
        max_restarts = 30 if target_n <= 15 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            n = target_n
            triples = triples_by_n[n]
            colors = [random.randint(0, 1) for _ in range(len(triples))]
            red_adj, blue_adj = build_adj(n, colors)
            
            # Local search with repair
            max_steps = 2000
            no_improve = 0
            
            for step in range(max_steps):
                if time.time() > deadline - 2:
                    break
                
                # Check validity
                if is_valid(n, colors, red_adj, blue_adj):
                    found = True
                    break
                
                # Find a violation
                r4 = find_red_k4(n, red_adj)
                b5 = find_blue_k5(n, blue_adj)
                
                if r4 is not None:
                    u, v, w1, w2 = r4
                    # Triples in the K4
                    candidates = [
                        tuple(sorted((u, v, w1))),
                        tuple(sorted((u, v, w2))),
                        tuple(sorted((u, w1, w2))),
                        tuple(sorted((v, w1, w2)))
                    ]
                    # Flip a random triple
                    flip_triple(n, colors, red_adj, blue_adj, random.choice(candidates))
                    no_improve = 0
                elif b5 is not None:
                    a, b, c, d, e = b5
                    # All 10 triples in the K5
                    candidates = []
                    for t in combinations([a, b, c, d, e], 3):
                        candidates.append(tuple(sorted(t)))
                    # Flip a random triple
                    flip_triple(n, colors, red_adj, blue_adj, random.choice(candidates))
                    no_improve = 0
                else:
                    break
            
            if found:
                best_n = n
                best_colors = colors
                break
        
        if found:
            # Try to go higher
            continue
        else:
            # Couldn't find valid coloring for this n, try next
            continue
    
    # If we didn't find anything, use fallback
    if best_n == 0:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}

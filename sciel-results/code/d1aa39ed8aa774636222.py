import random
import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def girth_check(adj, n, max_g):
        """Check if graph has girth >= max_g. Returns the girth found (or max_g if none shorter)."""
        for target in range(3, max_g):
            for v in range(n):
                seen = set()
                frontier = set()
                for nb in adj[v]:
                    frontier.add((nb, v))
                for length in range(1, target):
                    new_frontier = set()
                    for (u, p) in frontier:
                        for nb in adj[u]:
                            if nb == p:
                                continue
                            if nb == v:
                                return length + 1
                            if (nb, u) not in seen:
                                seen.add((nb, u))
                                new_frontier.add((nb, u))
                    frontier = new_frontier
                    if not frontier:
                        break
        return max_g

    def find_short_cycle(adj, n, max_len):
        """Find one cycle of length < max_len, return list of vertices or None."""
        for v in range(n):
            # BFS with parent tracking for shortest cycle back to v
            dist = {v: 0}
            parent = {v: None}
            q = deque([v])
            while q:
                u = q.popleft()
                if dist[u] >= max_len - 1:
                    continue
                for nb in adj[u]:
                    if nb == v:
                        # Found cycle v -> ... -> u -> v
                        cycle = [nb]
                        cur = u
                        while cur is not None and cur != v:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
                    if nb not in dist:
                        dist[nb] = dist[u] + 1
                        parent[nb] = u
                        q.append(nb)
        return None

    def make_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def cayley_3reg(p, a, b, c):
        """Build Cayley graph on Z_p with generators {±a, ±b, ±c}.
        Returns adj list or None if not valid (e.g., a+b+c != 0 mod p for degree 3)."""
        # For degree 3 on Z_p, we need exactly 3 distinct non-zero elements
        # such that each element's neighbors are x+a, x-b, x+c (or similar)
        # Actually for 3-regular: use generators {a, -a, b, -b, c, -c} but that's degree 6
        # For degree 3: use a directed Cayley-like or use {a, b, c} with a+b+c=0
        # Neighbors of x: x+a, x+b, x+c where a+b+c=0 mod p
        # This gives degree 3 (each x connects to x+a, x+b, x+c)
        # But we need undirected: edge x-(x+a) means x+a also has neighbor x, so x+a's neighbors are x+a+a, x+a+b, x+a+c
        # For this to be symmetric we need the generator set to be closed under negation
        # For degree 3: impossible with pure Cayley on abelian group (need even degree or use non-abelian)
        # Instead: use a "cubic" construction via two perfect matchings + one
        # Let's use: neighbors of x are (x+a)%p, (x-b)%p, (x+c)%p where a,b,c chosen
        # This is not necessarily symmetric. Let me think differently.
        # 
        # Better approach: construct using a known family.
        # For k=3, use the "cubic cage" approach:
        # Take p prime, and define neighbors as x -> x+s for s in {s1, s2, s3}
        # where s1+s2+s3 = 0 mod p, and we make it undirected by ensuring
        # that if x connects to x+s, then x+s connects to x (i.e., -s is also a generator)
        # But that gives degree 6. 
        #
        # Alternative: use a bipartite construction or a different approach.
        # Let's just return None and use a different method.
        return None

    def random_3reg(n):
        """Generate a random 3-regular graph on n vertices."""
        for _ in range(50):
            stubs = []
            for i in range(n):
                stubs += [i]*3
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3*n, 2):
                a, b = stubs[i], stubs[i+1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b); adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def repair_short_cycles(adj, n, target_girth, max_iter=2000):
        """Iteratively find and break short cycles via edge swaps."""
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            cycle = find_short_cycle(adj, n, target_girth)
            if cycle is None:
                return True
            
            # Try to break the cycle by swapping an edge on the cycle
            # with a non-cycle edge
            cycle_set = set(cycle)
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                if v in adj[u]:
                    cycle_edges.append((u, v))
            
            # Pick a random edge on the cycle and try to swap it
            swapped = False
            for _ in range(20):
                if not cycle_edges:
                    break
                ui, vi = random.choice(cycle_edges)
                # Find another edge to swap with
                candidates = []
                for w in range(n):
                    if w == ui or w == vi:
                        continue
                    for x in adj[w]:
                        if x in (ui, vi) or x == w:
                            continue
                        if x in adj[ui] or x in adj[vi]:
                            continue
                        candidates.append((w, x))
                
                if not candidates:
                    continue
                
                w, x = random.choice(candidates)
                
                # Perform swap: remove (ui,vi) and (w,x), add (ui,w) and (vi,x)
                # or (ui,x) and (vi,w)
                if random.random() < 0.5:
                    new_pairs = [(ui, w), (vi, x)]
                else:
                    new_pairs = [(ui, x), (vi, w)]
                
                # Check validity
                valid = True
                for a, b in new_pairs:
                    if b in adj[a]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Apply swap
                adj[ui].remove(vi); adj[vi].remove(ui)
                adj[w].remove(x); adj[x].remove(w)
                for a, b in new_pairs:
                    adj[a].append(b); adj[b].append(a)
                
                # Check if we improved
                new_cycle = find_short_cycle(adj, n, target_girth)
                if new_cycle is None:
                    swapped = True
                    break
                else:
                    # Revert
                    adj[ui].remove(new_pairs[0][1]); adj[new_pairs[0][1]].remove(ui)
                    adj[vi].remove(new_pairs[1][1]); adj[new_pairs[1][1]].remove(vi)
                    adj[ui].append(vi); adj[vi].append(ui)
                    adj[w].append(x); adj[x].append(w)
            
            if swapped:
                return True
            
            # If we couldn't break it, try a different approach: random swap
            a = random.randint(0, n-1)
            b = random.choice(adj[a])
            c = random.randint(0, n-1)
            if c in (a, b):
                continue
            d = random.choice(adj[c])
            if d in (a, b, c):
                continue
            if c in adj[a] or d in adj[b]:
                continue
            
            adj[a].remove(b); adj[b].remove(a)
            adj[c].remove(d); adj[d].remove(c)
            adj[a].append(c); adj[c].append(a)
            adj[b].append(d); adj[d].append(b)
            
            # Check if girth improved
            girth_now = girth_check(adj, n, target_girth)
            if girth_now >= target_girth:
                return True
            else:
                # Keep the change if it didn't make things worse
                # (we'll continue searching)
                pass
        
        return girth_check(adj, n, target_girth) >= target_girth

    # Main strategy:
    # 1. Try random 3-regular graphs at various sizes, repair short cycles
    # 2. Fall back to LUW construction
    
    best_n = None
    best_edges = None
    
    # Try sizes from small to large
    for n_try in [512, 514, 516, 520, 528, 540, 560, 600, 640, 700, 800, 900, 936]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            adj = random_3reg(n_try)
            if adj is None:
                continue
            
            # Check girth
            girth = girth_check(adj, n_try, g + 1)
            if girth >= g:
                edges = make_edges(adj, n_try)
                return n_try, edges
            
            # Try to repair
            if repair_short_cycles(adj, n_try, g, max_iter=500):
                edges = make_edges(adj, n_try)
                return n_try, edges
    
    # Fallback: LUW construction
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw

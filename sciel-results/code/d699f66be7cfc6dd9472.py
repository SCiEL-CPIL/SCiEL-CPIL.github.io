import random
import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    # Moore bound for k=3, g=16: 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    # So n must be >= 512 (even, since bipartite)
    
    def girth_check(adj, n, max_g):
        """Check if graph has girth >= max_g. Returns the girth found (or max_g if none shorter)."""
        for target in range(3, max_g):
            for v in range(n):
                # Non-backtracking walk enumeration
                # State: (current_node, prev_node)
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

    def make_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def random_3reg(n):
        """Generate a random 3-regular graph on n vertices."""
        for _ in range(100):
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

    def count_short_cycles(adj, n, max_g):
        """Count the number of cycles shorter than max_g (rough estimate via BFS)."""
        count = 0
        for target in range(4, max_g, 2):  # even cycles
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
                                count += 1
                                break
                            if (nb, u) not in seen:
                                seen.add((nb, u))
                                new_frontier.add((nb, u))
                    frontier = new_frontier
                    if not frontier:
                        break
                if count > 0:
                    break
            if count > 0:
                break
        return count

    # Strategy: Try to find a 3-regular graph of girth >= 16 with as few vertices as possible
    # Start from small n and increase
    
    best_result = None
    
    for n_try in [512, 514, 516, 520, 528, 540, 560, 600, 640, 700, 800, 900, 936]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            adj = random_3reg(n_try)
            if adj is None:
                continue
            
            # Quick girth check
            girth = girth_check(adj, n_try, 17)
            if girth >= 16:
                edges = make_edges(adj, n_try)
                return n_try, edges
            
            # Local search: try to increase girth via edge swaps
            current_girth = girth
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Pick two edges to swap
                a = random.randint(0, n_try-1)
                if len(adj[a]) < 3:
                    continue
                b = random.choice(adj[a])
                c = random.randint(0, n_try-1)
                if c in (a, b):
                    continue
                if len(adj[c]) < 3:
                    continue
                d = random.choice(adj[c])
                if d in (a, b, c):
                    continue
                if c in adj[a] or d in adj[b]:
                    continue
                
                # Perform swap
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                adj[a].append(c); adj[c].append(a)
                adj[b].append(d); adj[d].append(b)
                
                new_girth = girth_check(adj, n_try, 17)
                
                if new_girth >= 16:
                    edges = make_edges(adj, n_try)
                    return n_try, edges
                
                if new_girth > current_girth:
                    current_girth = new_girth
                else:
                    # Revert
                    adj[a].remove(c); adj[c].remove(a)
                    adj[b].remove(d); adj[d].remove(b)
                    adj[a].append(b); adj[b].append(a)
                    adj[c].append(d); adj[d].append(c)
    
    # Fallback: LUW construction
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

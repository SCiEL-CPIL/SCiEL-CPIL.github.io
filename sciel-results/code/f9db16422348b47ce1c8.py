import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 20
    num_colors = 20

    def check_valid(edges):
        if time.time() > deadline:
            return False
        adj = [[None]*n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        for a in range(n):
            for b in range(n):
                if b == a or adj[a][b] is None:
                    continue
                for c in range(n):
                    if c == a or c == b or adj[b][c] is None:
                        continue
                    for d in range(n):
                        if d == a or d == b or d == c:
                            continue
                        if adj[c][d] is None or adj[d][a] is None:
                            continue
                        if len({adj[a][b], adj[b][c], adj[c][d], adj[d][a]}) != 4:
                            return False
        return True

    # Try simple structured colourings
    for func in [lambda u,v:(u+v)%20, lambda u,v:(u-v)%20, lambda u,v:(u*v)%20,
                 lambda u,v:(u^v)%20, lambda u,v:(3*u+v)%20, lambda u,v:(u+3*v)%20]:
        if time.time() > deadline - 40:
            break
        edges = [[u,v,func(u,v)] for u in range(n) for v in range(u+1,n)]
        if check_valid(edges):
            return edges

    # Random search with local improvement
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        random.seed(attempt * 104729 + 7919)
        edges = [[u,v,random.randint(0,19)] for u in range(n) for v in range(u+1,n)]
        
        for it in range(200):
            if time.time() > deadline - 5:
                break
            if check_valid(edges):
                return edges
            # Find a violation and recolor
            adj = [[None]*n for _ in range(n)]
            for u,v,c in edges:
                adj[u][v] = c
                adj[v][u] = c
            fixed = False
            for a in range(n):
                if fixed: break
                for b in range(n):
                    if fixed or b==a or adj[a][b] is None: continue
                    for c in range(n):
                        if fixed or c in (a,b) or adj[b][c] is None: continue
                        for d in range(n):
                            if fixed or d in (a,b,c): continue
                            if adj[c][d] is None or adj[d][a] is None: continue
                            if len({adj[a][b],adj[b][c],adj[c][d],adj[d][a]}) != 4:
                                # Recolor edge (a,b)
                                for i,e in enumerate(edges):
                                    if (e[0],e[1])==(a,b) or (e[0],e[1])==(b,a):
                                        edges[i][2] = random.randint(0,19)
                                        fixed = True
                                        break
                                break
            if not fixed:
                break

    return [[0, v, 0] for v in range(1, n)]

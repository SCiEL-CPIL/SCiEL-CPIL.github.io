"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 125
    
    n = 16
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    
    def build_grotzsch():
        adj = [[0]*n for _ in range(n)]
        edges = []
        for u, w in cycle:
            adj[u][w] = 1; adj[w][u] = 1
            edges.append([min(u,w), max(u,w)])
        for idx, (u, w) in enumerate(cycle):
            v = 5 + idx
            adj[u][v] = 1; adj[v][u] = 1
            adj[w][v] = 1; adj[v][w] = 1
            edges.append([min(u,v), max(u,v)])
            edges.append([min(w,v), max(w,v)])
        for u in range(5, 10):
            adj[u][10] = 1; adj[10][u] = 1
            edges.append([min(u,10), max(u,10)])
        return adj, edges
    
    def can_add(adj, i, j):
        if adj[i][j]:
            return False
        for k in range(n):
            if k != i and k != j and adj[i][k] and adj[j][k]:
                return False
        return True
    
    def greedy(adj, edges, order):
        for i, j in order:
            if time.time() - start_time > DEADLINE:
                break
            if can_add(adj, i, j):
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([min(i,j), max(i,j)])
        return edges
    
    adj0, edges0 = build_grotzsch()
    missing = [(i,j) for i in range(n) for j in range(i+1, n) if not adj0[i][j]]
    
    # Multiple orderings
    orderings = []
    
    # 1. Natural order
    orderings.append(sorted(missing))
    
    # 2. Isolated vertices first
    orderings.append(sorted(missing, key=lambda e: (min(e)>=11, max(e)>=11, e[0], e[1])))
    
    # 3. By degree sum descending
    def deg_sum(e):
        return -(sum(adj0[e[0]]) + sum(adj0[e[1]]))
    orderings.append(sorted(missing, key=deg_sum))
    
    # 4. Reverse natural
    orderings.append(sorted(missing, key=lambda e: (-e[0], -e[1])))
    
    # 5. Hash-based
    orderings.append(sorted(missing, key=lambda e: (e[0]*7+e[1]*11)%29))
    
    # 6. Isolated pairs first
    orderings.append(sorted(missing, key=lambda e: (min(e)<11, max(e)<11, e[0], e[1])))
    
    # 7. By distance
    orderings.append(sorted(missing, key=lambda e: (abs(e[0]-e[1]), e[0])))
    
    # 8. Random-ish
    orderings.append(sorted(missing, key=lambda e: (e[0]*13+e[1]*17)%31))
    
    best = list(edges0)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        adj, edges = build_grotzsch()
        edges = greedy(adj, edges, order)
        if len(edges) > len(best):
            best = list(edges)
    
    return best

# EVOLVE-BLOCK-END

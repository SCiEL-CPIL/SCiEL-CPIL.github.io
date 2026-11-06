import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    def is_valid_graph(adj, n):
        """Check no K5 and no independent set of size 5"""
        # Check for K5
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if not adj[a][d] or not adj[b][d] or not adj[c][d]:
                            continue
                        for e in range(d+1, n):
                            if adj[a][e] and adj[b][e] and adj[c][e] and adj[d][e]:
                                return False
        # Check for independent set of size 5
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if adj[a][c] or adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] or adj[b][d] or adj[c][d]:
                            continue
                        for e in range(d+1, n):
                            if not adj[a][e] and not adj[b][e] and not adj[c][e] and not adj[d][e]:
                                return False
        return True
    
    # Try 42-vertex Exoo graph using circulant construction
    # (42, 20, 9, 10) strongly regular graph
    n42 = 42
    adj42 = [[False] * n42 for _ in range(n42)]
    
    # Connection set for the circulant (42,20,9,10) SRG
    # Each vertex i connected to i+d mod 42 for d in conn_set
    # Need 10 differences (each gives 2 edges per vertex, total degree 20)
    conn_set_42 = {1, 2, 3, 4, 5, 6, 7, 8, 11, 14}
    
    for i in range(n42):
        for d in conn_set_42:
            j = (i + d) % n42
            adj42[i][j] = True
    
    if is_valid_graph(adj42, n42):
        edge_list = []
        for i in range(n42):
            for j in range(i + 1, n42):
                if adj42[i][j]:
                    edge_list.append((i, j))
        return n42, edge_list
    
    # Fallback: try other connection sets for 42 vertices
    # Try different connection sets
    possible_conn_sets = [
        {1, 2, 3, 4, 5, 6, 7, 8, 11, 15},
        {1, 2, 3, 4, 5, 6, 7, 8, 12, 14},
        {1, 2, 3, 4, 5, 6, 7, 8, 11, 13},
        {1, 2, 3, 4, 5, 6, 7, 8, 10, 14},
        {1, 2, 3, 4, 5, 6, 7, 8, 11, 16},
        {1, 2, 3, 4, 5, 6, 7, 8, 12, 15},
        {1, 2, 3, 4, 5, 6, 7, 8, 13, 14},
        {1, 2, 3, 4, 5, 6, 7, 8, 10, 15},
        {1, 2, 3, 4, 5, 6, 7, 8, 10, 13},
        {1, 2, 3, 4, 5, 6, 7, 8, 12, 13},
    ]
    
    for conn_set in possible_conn_sets:
        adj = [[False] * n42 for _ in range(n42)]
        for i in range(n42):
            for d in conn_set:
                j = (i + d) % n42
                adj[i][j] = True
        if is_valid_graph(adj, n42):
            edge_list = []
            for i in range(n42):
                for j in range(i + 1, n42):
                    if adj[i][j]:
                        edge_list.append((i, j))
            return n42, edge_list
    
    # Fallback to Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edge_list.append((i, j))
    
    return n, edge_list

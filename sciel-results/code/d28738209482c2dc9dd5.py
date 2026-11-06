import random
from itertools import combinations

def construct():
    best_edges = 0
    best_edge_list = None
    
    def has_k33(adj):
        for tri1 in combinations(range(20), 3):
            common = [v for v in range(20) if v not in tri1 and adj[tri1[0]][v] and adj[tri1[1]][v] and adj[tri1[2]][v]]
            if len(common) >= 3:
                return True
        return False
    
    for seed in range(50):
        random.seed(seed)
        adj = [[False]*20 for _ in range(20)]
        for i in range(20):
            for j in range(i+1, 20):
                if random.random() < 0.6:
                    adj[i][j] = adj[j][i] = True
        
        # Remove K3,3s
        changed = True
        while changed:
            changed = False
            for tri1 in combinations(range(20), 3):
                common = [v for v in range(20) if v not in tri1 and adj[tri1[0]][v] and adj[tri1[1]][v] and adj[tri1[2]][v]]
                if len(common) >= 3:
                    tri2 = list(combinations(common, 3))[0]
                    a = random.choice(tri1)
                    b = random.choice(tri2)
                    adj[a][b] = adj[b][a] = False
                    changed = True
                    break
        
        # Try to add edges back
        for i in range(20):
            for j in range(i+1, 20):
                if not adj[i][j]:
                    adj[i][j] = adj[j][i] = True
                    if has_k33(adj):
                        adj[i][j] = adj[j][i] = False
        
        edge_count = sum(1 for i in range(20) for j in range(i+1, 20) if adj[i][j])
        if edge_count > best_edges:
            best_edges = edge_count
            best_edge_list = [[i, j] for i in range(20) for j in range(i+1, 20) if adj[i][j]]
    
    return best_edge_list if best_edge_list else []

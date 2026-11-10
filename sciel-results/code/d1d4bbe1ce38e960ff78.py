import random
from itertools import combinations

def construct():
    n = 20
    
    # Use a deterministic construction based on a strongly regular graph
    # We'll use the complement of the Petersen graph blown up, or use a known construction
    # Actually, let's use a greedy approach with a good initial structure
    
    # Start with a random dense graph and use local search to remove edges
    # to satisfy the K3,3-free constraint
    
    best_edges = []
    best_count = 0
    
    # Multiple random restarts
    for seed in range(100):
        random.seed(seed)
        
        # Start with a complete graph
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = True
                adj[j][i] = True
        
        # Count edges
        def count_edges():
            return sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        # Check if graph is K3,3-free
        def is_k33_free():
            for a, b, c in combinations(range(n), 3):
                common = 0
                for x in range(n):
                    if x not in (a, b, c) and adj[a][x] and adj[b][x] and adj[c][x]:
                        common += 1
                        if common >= 3:
                            return False
            return True
        
        # Find a violation
        def find_violation():
            for a, b, c in combinations(range(n), 3):
                common = []
                for x in range(n):
                    if x not in (a, b, c) and adj[a][x] and adj[b][x] and adj[c][x]:
                        common.append(x)
                        if len(common) >= 3:
                            return (a, b, c, common)
            return None
        
        # Local search: remove edges to fix violations
        for _ in range(2000):
            viol = find_violation()
            if viol is None:
                break
            
            a, b, c, common = viol
            # Remove an edge incident to one of the common neighbors
            # Try removing edges that appear in many violations
            # Simple heuristic: remove a random edge from the common neighbors
            x = random.choice(common)
            # Remove one of the edges (x,a), (x,b), or (x,c)
            choice = random.choice([a, b, c])
            adj[x][choice] = False
            adj[choice][x] = False
        
        # Try to add back edges
        for _ in range(100):
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        # Try adding edge (i, j)
                        adj[i][j] = True
                        adj[j][i] = True
                        if is_k33_free():
                            added = True
                        else:
                            adj[i][j] = False
                            adj[j][i] = False
            if not added:
                break
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        best_edges.append([i, j])
    
    return best_edges

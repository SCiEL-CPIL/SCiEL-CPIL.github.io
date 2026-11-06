import time

def construct():
    # Try Paley 41 first
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    if _is_55_free(q, edges):
        return q, edges
    
    # Fallback to Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

def _is_55_free(n, edges):
    # Check for K5 and Independent Set of size 5
    adj = [[False] * n for _ in range(n)]
    for i, j in edges:
        adj[i][j] = True
        adj[j][i] = True
        
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]: continue
            for k in range(j+1, n):
                if not adj[i][k] or not adj[j][k]: continue
                for l in range(k+1, n):
                    if not adj[i][l] or not adj[j][l] or not adj[k][l]: continue
                    for m in range(l+1, n):
                        if adj[i][m] and adj[j][m] and adj[k][m] and adj[l][m]:
                            return False
                            
    # Check Independent Set of size 5
    # Independent set in G is a clique in complement G
    # Complement adjacency
    comp_adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp_adj[i][j] = True
                comp_adj[j][i] = True
                
    # Check K5 in complement
    for i in range(n):
        for j in range(i+1, n):
            if not comp_adj[i][j]: continue
            for k in range(j+1, n):
                if not comp_adj[i][k] or not comp_adj[j][k]: continue
                for l in range(k+1, n):
                    if not comp_adj[i][l] or not comp_adj[j][l] or not comp_adj[k][l]: continue
                    for m in range(l+1, n):
                        if comp_adj[i][m] and comp_adj[j][m] and comp_adj[k][m] and comp_adj[l][m]:
                            return False
                            
    return True

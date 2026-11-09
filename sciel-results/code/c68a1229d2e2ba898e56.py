import sys
import time

def construct():
    """
    Constructs a graph on 15 vertices maximizing edges such that
    no 3-edge path has endpoints of equal degree.
    
    Strategy:
    1. Start with a complete bipartite graph K_{7,8}. 
       Vertices 0-6 form set A, vertices 7-14 form set B.
       Edges exist between all pairs in A and B.
       Degrees: A vertices have degree 8, B vertices have degree 7.
       This is valid because any 3-path s-u-v-t must have s,t in different parts 
       (since no edges within parts), so d(s) != d(t) because 8 != 7.
    2. Add one edge within set A, e.g., (0, 1).
       New degrees: d(0)=9, d(1)=9, other A=8, B=7.
       Verification:
       - Endpoints with degree 9 (0,1): No 3-path connects them because 
         any path 0-u-v-1 requires u in B, v in A. 
         1 is only connected to 0 in A, so v must be 0, making path non-simple.
       - Endpoints with degree 8 (A\{0,1}): No 3-path connects them because 
         any path s-u-v-t with s,t in A\{0,1} requires v in A connected to t.
         The only edges in A are (0,1), so v must be 0 or 1.
         If v=0, t must be 1 (deg 9, mismatch). If v=1, t must be 0 (deg 9, mismatch).
       - Endpoints with degree 7 (B): No 3-path connects them because 
         any path s-u-v-t with s,t in B requires v in B connected to t.
         No edges in B.
       
       Total edges: 7*8 + 1 = 57.
    """
    n = 15
    edges = []
    
    # Set A: 0 to 6
    # Set B: 7 to 14
    set_a = list(range(0, 7))
    set_b = list(range(7, 15))
    
    # Add all bipartite edges
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    # Add one edge within set A
    edges.append([0, 1])
    
    # Ensure types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result

if __name__ == "__main__":
    # Optional: verification for local testing
    def verify(edges, n):
        adj = [[False]*n for _ in range(n)]
        deg = [0]*n
        for u, v in edges:
            if adj[u][v]:
                return False
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            
        # Check 3-paths
        # s-u-v-t
        for u in range(n):
            for v in range(n):
                if u == v or not adj[u][v]:
                    continue
                # s is neighbor of u, not v
                # t is neighbor of v, not u
                # s != t
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj[t][v]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    res = construct()
    print(f"Edges: {len(res)}")
    print(f"Valid: {verify(res, 15)}")

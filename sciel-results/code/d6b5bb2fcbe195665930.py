"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 15}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = P['n']
    # Construct K_{7,8}
    # Partitions A (0..6) and B (7..14)
    A = list(range(7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Verify validity (optional but good for safety)
    # Check degrees
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
        
    # Check all 3-paths
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    valid = True
    # Iterate over all pairs (s, t) with s != t
    # If deg(s) == deg(t), check if there is a path of length 3
    # Path s-u-v-t implies s~u, u~v, v~t, s!=t, s!=v, u!=t
    # Since graph is simple, s~u and u~v implies s!=v unless s=v which is not allowed in simple path definition usually, but "simple path" means distinct vertices.
    # So s, u, v, t must be distinct.
    
    # Optimization: Only check pairs with equal degree
    deg_to_vertices = {}
    for i in range(n):
        deg_to_vertices.setdefault(degrees[i], []).append(i)
        
    for deg, verts in deg_to_vertices.items():
        if len(verts) < 2:
            continue
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                s = verts[i]
                t = verts[j]
                # Check if path s-u-v-t exists
                found_path = False
                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t:
                            continue
                        if t in adj[v]:
                            found_path = True
                            break
                    if found_path:
                        break
                if found_path:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid sparse graph if K_{7,8} somehow fails (it shouldn't)
        return [[0, v] for v in range(1, n)]
        
    return edges

# EVOLVE-BLOCK-END

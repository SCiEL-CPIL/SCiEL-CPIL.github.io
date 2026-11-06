"""Dense edge-coloured graph on 20 vertices with every 4-cycle rainbow."""
# EVOLVE-BLOCK-START

def construct():
    n = 20
    # Construct a C4-free graph using the polarity graph of the projective 
    # plane of order 4 (which has 21 points), take 20 of them.
    # 
    # Projective plane of order 4: 21 points, 21 lines, each line has 5 points.
    # Polarity graph: connect point p to line L if p is on L. This is a 
    # (21, 5)-regular bipartite graph, but as a single graph (identifying 
    # points and lines), it's a C4-free graph on 21 vertices.
    # 
    # Simpler: use a known explicit C4-free graph on 20 vertices.
    # Let's use the greedy approach but verify it gives a good count.
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Greedy addition in natural order
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            ok = True
            for a in common:
                for b in common:
                    if a < b and b in adj[a]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v, len(edges) % 20])
    
    return edges
# EVOLVE-BLOCK-END

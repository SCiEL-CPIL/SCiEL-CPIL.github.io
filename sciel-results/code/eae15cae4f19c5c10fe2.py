"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 20
    ncolors = 20
    
    # Try to build a dense graph with valid coloring
    # Strategy: Use a bipartite graph K_{10,10} with structured coloring
    # Vertices 0-9 on left, 10-19 on right
    
    left = list(range(10))
    right = list(range(10, 20))
    
    # Color edge (i, j) where i in left, j in right
    # Use color = (i * 2 + (j - 10)) % 20
    # This ensures for any 4-cycle with left vertices i1,i2 and right vertices j1,j2:
    # colors are (2*i1 + j1) % 20, (2*i1 + j2) % 20, (2*i2 + j1) % 20, (2*i2 + j2) % 20
    # For these to be distinct, we need no two to be equal mod 20
    
    edges = []
    for i in left:
        for j in right:
            c = (i * 2 + (j - 10)) % ncolors
            edges.append([i, j, c])
    
    # Verify: check all 4-cycles
    # In bipartite graph, 4-cycle: i1, i2 in left; j1, j2 in right
    # Edges: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
    
    # Build edge lookup
    edge_color = {}
    for u, v, c in edges:
        edge_color[(min(u,v), max(u,v))] = c
    
    valid = True
    for i1 in left:
        for i2 in left:
            if i1 >= i2:
                continue
            for j1 in right:
                for j2 in right:
                    if j1 >= j2:
                        continue
                    c1 = edge_color.get((i1, j1))
                    c2 = edge_color.get((i1, j2))
                    c3 = edge_color.get((i2, j1))
                    c4 = edge_color.get((i2, j2))
                    if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
                        if len({c1, c2, c3, c4}) != 4:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return edges
    
    # If bipartite construction doesn't work, try greedy approach
    # Start with complete graph and try to color it, removing edges if needed
    
    # Actually, let's try a different coloring scheme
    # color = (i + j) % 20 won't work
    # color = (i * 3 + j) % 20
    edges2 = []
    for i in left:
        for j in right:
            c = (i * 3 + (j - 10)) % ncolors
            edges2.append([i, j, c])
    
    edge_color2 = {}
    for u, v, c in edges2:
        edge_color2[(min(u,v), max(u,v))] = c
    
    valid2 = True
    for i1 in left:
        for i2 in left:
            if i1 >= i2:
                continue
            for j1 in right:
                for j2 in right:
                    if j1 >= j2:
                        continue
                    c1 = edge_color2.get((i1, j1))
                    c2 = edge_color2.get((i1, j2))
                    c3 = edge_color2.get((i2, j1))
                    c4 = edge_color2.get((i2, j2))
                    if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
                        if len({c1, c2, c3, c4}) != 4:
                            valid2 = False
                            break
                    if time.time() > deadline:
                        return edges  # Return what we have
                if not valid2:
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if valid2:
        return edges2
    
    # Fallback: try adding edges to a smaller base graph
    # Use a random greedy approach
    random.seed(42)
    
    # Start with no edges, try to add all possible edges with valid coloring
    # This is complex, so just return the best bipartite result we found
    # K_{10,10} has 100 edges, which is much better than 19
    
    # Try yet another coloring: color = (i * 10 + (j-10)) % 20
    edges3 = []
    for i in left:
        for j in right:
            c = (i * 10 + (j - 10)) % ncolors
            edges3.append([i, j, c])
    
    edge_color3 = {}
    for u, v, c in edges3:
        edge_color3[(min(u,v), max(u,v))] = c
    
    valid3 = True
    for i1 in left:
        for i2 in left:
            if i1 >= i2:
                continue
            for j1 in right:
                for j2 in right:
                    if j1 >= j2:
                        continue
                    c1 = edge_color3.get((i1, j1))
                    c2 = edge_color3.get((i1, j2))
                    c3 = edge_color3.get((i2, j1))
                    c4 = edge_color3.get((i2, j2))
                    if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
                        if len({c1, c2, c3, c4}) != 4:
                            valid3 = False
                            break
                    if time.time() > deadline:
                        return edges3
                if not valid3:
                    break
            if not valid3:
                break
        if not valid3:
            break
    
    if valid3:
        return edges3
    
    # Return the star as fallback (always valid)
    return [[0, v, 0] for v in range(1, 20)]
# EVOLVE-BLOCK-END

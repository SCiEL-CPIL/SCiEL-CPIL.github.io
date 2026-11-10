import time

def construct():
    # Generate points of P^2(F_5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    cx = 1
                    cy = (y * inv_x) % 5
                    cz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    cx = 0
                    cy = 1
                    cz = (z * inv_y) % 5
                else:
                    cx = 0
                    cy = 0
                    cz = 1
                points.append((cx, cy, cz))
    
    # Polarity defined by symmetric matrix A
    # A = [[1, 0, 0], [0, 1, 1], [0, 1, 0]]
    # u^T A v = u0*v0 + u1*v1 + u1*v2 + u2*v1
    def dot(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[1]*v[2] + u[2]*v[1]) % 5

    edges = []
    edge_set = set()
    
    n = len(points)
    # Construct the 90-edge polarity graph
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                edge_set.add((i, j))
    
    # Add one edge to reach 91 edges
    # Any non-edge in this graph creates exactly 1 C4
    added = False
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                edges.append([i, j])
                added = True
                break
        if added:
            break
            
    return edges

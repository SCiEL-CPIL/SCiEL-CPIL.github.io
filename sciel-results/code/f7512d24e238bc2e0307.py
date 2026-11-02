import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) / 2 * (2 * r + q)
        return x, y

    # Define candidate clusters in axial coordinates
    # We want compact clusters.
    
    # Cluster 1: 7-core + 4 extending in a "fan"
    c1 = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,-1), (2,-2), (1,-2), (0,-2)
    ]

    # Cluster 2: 7-core + 4 extending in two opposite "fans"
    c2 = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,-1), (2,-2), (-2,1), (-2,2)
    ]

    # Cluster 3: 7-core + 4 forming a "cap"
    c3 = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (0,2), (1,1), (-1,2), (2,0)
    ]
    
    # Cluster 4: A 3x3 block (9) + 2
    # 3x3 block in axial:
    # (-1,0), (0,0), (1,0)
    # (-1,1), (0,1), (1,1)
    # (-1,2), (0,2), (1,2)
    # This is 9 hexes.
    # Add 2 more: (2,1), (2,2)
    c4 = [
        (-1,0), (0,0), (1,0),
        (-1,1), (0,1), (1,1),
        (-1,2), (0,2), (1,2),
        (2,1), (2,2)
    ]

    candidates = [c1, c2, c3, c4]
    
    # Normals for flat-top hexagon sides
    normals = []
    for i in range(6):
        angle = np.deg2rad(30 + i * 60)
        normals.append((np.cos(angle), np.sin(angle)))
        
    h = np.sqrt(3) / 2
    
    best_L = float('inf')
    best_inner = []
    best_center = (0.0, 0.0)
    
    for cand in candidates:
        points = np.array([axial_to_cartesian(q, r) for q, r in cand])
        center = np.mean(points, axis=0)
        rel_points = points - center
        
        max_proj = 0
        for nx, ny in normals:
            projs = rel_points[:, 0] * nx + rel_points[:, 1] * ny
            max_abs = np.max(np.abs(projs))
            if max_abs > max_proj:
                max_proj = max_abs
                
        L = 1 + max_proj / h
        
        if L < best_L:
            best_L = L
            best_inner = [[x, y, 0.0] for x, y in points]
            best_center = (center[0], center[1])

    return best_inner, best_center, best_L, 0.0
# EVOLVE-BLOCK-END

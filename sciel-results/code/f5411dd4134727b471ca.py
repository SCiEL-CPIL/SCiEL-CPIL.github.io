import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Distance between centers of touching unit hexagons
    d = np.sqrt(3)
    
    # Define candidate topologies as lists of (q, r) lattice coordinates
    # Topology 1: 3x4 parallelogram
    top1 = [(q, r) for q in range(3) for r in range(4)]
    
    # Topology 2: 2x6 parallelogram
    top2 = [(q, r) for q in range(2) for r in range(6)]
    
    # Topology 3: Central 7 + 5 extensions
    # Center (0,0)
    # Ring: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Add 5 more: (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2) -- Just a guess, let's try to make it compact
    top3 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), 
            (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2)]
            
    # Topology 4: A more symmetric "hexagon" approximation
    # 1 center, 6 ring, 5 more on ring
    # Let's try adding to 5 of the 6 ring positions
    top4 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
            (2,0), (2,-1), (1,-2), (0,-2), (-1,-1)]
            
    candidates = [top1, top2, top3, top4]
    
    best_score = float('inf')
    best_inner = None
    best_side = None
    best_angle = 0.0
    
    # Pre-compute local vertices for each candidate
    # Local vertices of a unit hexagon centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    angles_local = np.deg2rad(np.arange(6) * 60)
    v_local = np.column_stack((np.cos(angles_local), np.sin(angles_local)))
    
    for topo in candidates:
        # Convert lattice coords to Cartesian
        # x = d * (q + r/2)
        # y = d * (sqrt(3)/2 * r) = 1.5 * r  (since d=sqrt(3), sqrt(3)/2 * sqrt(3) = 3/2 = 1.5)
        centers = np.array([
            (d * (q + r/2), 1.5 * r) for q, r in topo
        ])
        
        # Center of mass
        com = np.mean(centers, axis=0)
        centers = centers - com
        
        # Generate all vertices for this cluster
        # For each center, add v_local
        # Shape: (12, 6, 2)
        verts = centers[:, np.newaxis, :] + v_local[np.newaxis, :, :]
        # Flatten to (72, 2)
        verts = verts.reshape(-1, 2)
        
        # Function to compute S(theta)
        def calc_S(theta):
            c = np.cos(theta)
            s = np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rotated_verts = verts @ R.T
            
            # Outer hexagon normals (fixed)
            # Normals at 0, 60, 120, 180, 240, 300
            n_angles = np.deg2rad(np.arange(6) * 60)
            normals = np.column_stack((np.cos(n_angles), np.sin(n_angles)))
            
            # Projections
            # (72, 2) @ (2, 6) -> (72, 6)
            projs = rotated_verts @ normals.T
            max_proj = np.max(projs)
            
            # Side length S = 2 * max_proj / sqrt(3)
            return 2 * max_proj / np.sqrt(3)
        
        # Minimize calc_S(theta)
        # Use a grid search first, then refine
        thetas = np.linspace(0, 2*np.pi, 100)
        S_vals = np.array([calc_S(t) for t in thetas])
        idx_min = np.argmin(S_vals)
        
        # Refine around idx_min
        # Ternary search or just fine grid
        t_left = thetas[max(0, idx_min-1)]
        t_right = thetas[min(len(thetas)-1, idx_min+1)]
        
        # Fine grid search
        fine_thetas = np.linspace(t_left, t_right, 100)
        fine_S_vals = np.array([calc_S(t) for t in fine_thetas])
        idx_fine = np.argmin(fine_S_vals)
        
        optimal_theta = fine_thetas[idx_fine]
        optimal_S = fine_S_vals[idx_fine]
        
        if optimal_S < best_score:
            best_score = optimal_S
            best_angle = np.degrees(optimal_theta)
            # Rotate centers and prepare inner list
            c = np.cos(optimal_theta)
            s = np.sin(optimal_theta)
            R = np.array([[c, -s], [s, c]])
            rotated_centers = centers @ R.T
            
            inner = []
            for rc in rotated_centers:
                inner.append([rc[0], rc[1], best_angle])
            best_inner = inner
            best_side = optimal_S
            
    return best_inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

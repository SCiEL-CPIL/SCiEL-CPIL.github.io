import numpy as np

def hexagon_packing_11():
    # Initial configuration: 7-hex flower + 4 in gaps
    # Centers
    # Inner 7
    centers = [
        [0, 0],
        [np.sqrt(3)*np.cos(np.pi/6), np.sqrt(3)*np.sin(np.pi/6)], # 30
        [np.sqrt(3)*np.cos(np.pi/2), np.sqrt(3)*np.sin(np.pi/2)], # 90
        [np.sqrt(3)*np.cos(5*np.pi/6), np.sqrt(3)*np.sin(5*np.pi/6)], # 150
        [np.sqrt(3)*np.cos(7*np.pi/6), np.sqrt(3)*np.sin(7*np.pi/6)], # 210
        [np.sqrt(3)*np.cos(3*np.pi/2), np.sqrt(3)*np.sin(3*np.pi/2)], # 270
        [np.sqrt(3)*np.cos(11*np.pi/6), np.sqrt(3)*np.sin(11*np.pi/6)], # 330
    ]
    # Outer 4 in gaps (angles 60, 120, 240, 300)
    # Distance 3, angles 60, 120, 240, 300
    outer_angles = [np.pi/3, 2*np.pi/3, 4*np.pi/3, 5*np.pi/3]
    for ang in outer_angles:
        centers.append([3*np.cos(ang), 3*np.sin(ang)])
    
    centers = np.array(centers)
    angles = np.zeros(11) # All 0 deg
    
    # Simple optimization to reduce bounding box
    # We will try to minimize the max projection onto the 6 container normals
    
    def get_vertices(c, a):
        # c: (x,y), a: angle
        verts = []
        for i in range(6):
            ang = a + i * np.pi / 3
            verts.append([c[0] + np.cos(ang), c[1] + np.sin(ang)])
        return np.array(verts)

    def calc_side_length(centers, angles):
        all_verts = []
        for i in range(len(centers)):
            all_verts.append(get_vertices(centers[i], angles[i]))
        all_verts = np.vstack(all_verts)
        
        # Container normals at 30, 90, 150, 210, 270, 330
        normals = []
        for i in range(6):
            ang = np.pi/6 + i * np.pi/3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0
        for n in normals:
            projs = all_verts @ n
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        # Inradius = max_proj
        # Side = Inradius * 2 / sqrt(3)
        return max_proj * 2 / np.sqrt(3)

    # Local search
    best_side = calc_side_length(centers, angles)
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    for _ in range(100): # Few iterations to be safe
        # Random perturb
        trial_centers = best_centers + np.random.normal(0, 0.01, best_centers.shape)
        trial_angles = best_angles + np.random.normal(0, 0.01, best_angles.shape)
        
        # Check overlap (simple distance check for centers, not exact)
        # For unit hexes, min center dist is sqrt(3)
        valid = True
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(trial_centers[i] - trial_centers[j])
                if d < np.sqrt(3) - 1e-6:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            side = calc_side_length(trial_centers, trial_angles)
            if side < best_side:
                best_side = side
                best_centers = trial_centers
                best_angles = trial_angles

    inner_hex_data = np.column_stack((best_centers, best_angles * 180 / np.pi))
    outer_hex_data = np.array([0, 0, 0])
    return inner_hex_data, outer_hex_data, best_side

"""
Topology-Driven Constraint Satisfaction for Hexagon Packing.

Strategy:
- Define a contact graph (tree) for 11 unit hexagons.
- Parameterize the configuration by joint angles and rotation offsets.
- Use Nelder-Mead to minimize the outer hexagon side length.
- Enforce non-overlap via SAT (Separating Axis Theorem) penalty.
- Key innovation: Treat the packing as a rigid cluster with internal degrees of freedom
  that allow "sliding" at contacts, decoupling from fixed lattice assumptions.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# Contact graph: edges (parent, child)
# 0 is center.
# 1-6 ring around 0.
# 7-10 outer.
# Topology:
# 0 connected to 1..6
# 1 connected to 2 (via 0? no, direct contact)
# Let's define a spanning tree for kinematic chain.
# Root: 0
# Children of 0: 1, 2, 3, 4, 5, 6
# But 1 and 2 touch. To allow sliding, we make one of them a child of the other?
# If we use a tree, we lose the constraint that 1 and 2 must be at specific relative angle.
# Instead, we use a "soft" constraint: penalize deviation from contact distance for non-tree edges,
# but hard-constrain tree edges?
# No, better: Parameterize by relative angles for ALL contacts, but solve for consistency.
# Simpler: Use a tree topology for positions, and penalize overlaps for all pairs.
# Tree:
# 0 -> 1
# 1 -> 2
# 2 -> 3
# 3 -> 4
# 4 -> 5
# 5 -> 6
# 6 -> 0 (cycle? no, tree must be acyclic)
# Let's use:
# 0 is root.
# 1 is child of 0.
# 2 is child of 1.
# 3 is child of 2.
# 4 is child of 3.
# 5 is child of 4.
# 6 is child of 5.
# This creates a chain 0-1-2-3-4-5-6.
# Then 7 is child of 2 (touching 1 and 2).
# 8 is child of 4 (touching 3 and 4).
# 9 is child of 5 (touching 4 and 5).
# 10 is child of 1 (touching 6 and 1).
# This tree allows the ring to "breathe".

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (2, 7),  # 7 touches 1 and 2, parent is 2
    (4, 8),  # 8 touches 3 and 4, parent is 4
    (5, 9),  # 9 touches 4 and 5, parent is 5
    (1, 10)  # 10 touches 6 and 1, parent is 1
]

# Contact pairs for penalty (all pairs that should be touching)
CONTACTS = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
    (1, 10), (2, 7), (3, 8), (4, 8), (4, 9), (5, 9), (6, 10),
    (2, 1), # duplicate, keep unique
    (1, 7), # 7 touches 1
    (3, 7), # 7 touches 3? No, 7 is at 30 deg. 1 at 0, 2 at 60. 7 touches 1 and 2.
    # Let's list all intended contacts explicitly
]
# Correct contact list:
CONTACTS = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
    (1, 7), (2, 7),
    (3, 8), (4, 8),
    (4, 9), (5, 9),
    (1, 10), (6, 10)
]

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    sqrt3 = np.sqrt(3)
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_overlap_fast(v1, v2, tol=1e-7):
    """SAT check for two convex polygons."""
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _compute_positions(params):
    """
    Compute positions and angles from parameters.
    params: [joint_angles (10), rot_offsets (11)]
    joint_angles[i] is the angle of the vector from parent to child relative to the parent's local frame.
    """
    n = N
    joint_angles = params[:n-1]
    rot_offsets = params[n-1:]
    
    pos = np.zeros((n, 2))
    ang = np.zeros(n)
    
    # Root is 0.
    # ang[0] = rot_offsets[0]
    ang[0] = rot_offsets[0]
    pos[0] = [0.0, 0.0]
    
    # Topological order of children
    # We need to process parents before children.
    # Tree:
    # 0 -> 1
    # 1 -> 2, 10
    # 2 -> 3, 7
    # 3 -> 4
    # 4 -> 5, 8
    # 5 -> 6, 9
    
    children = {i: [] for i in range(n)}
    for p, c in EDGES:
        children[p].append(c)
        
    # BFS
    queue = [0]
    processed = set()
    processed.add(0)
    while queue:
        p = queue.pop(0)
        for c in children[p]:
            if c in processed:
                continue
            processed.add(c)
            
            # Determine which joint angle index corresponds to edge (p, c)
            # EDGES list order:
            # (0,1): idx 0
            # (1,2): idx 1
            # (2,3): idx 2
            # (3,4): idx 3
            # (4,5): idx 4
            # (5,6): idx 5
            # (2,7): idx 6
            # (4,8): idx 7
            # (5,9): idx 8
            # (1,10): idx 9
            
            # Map edge to index
            edge_idx = None
            for i, (ep, ec) in enumerate(EDGES):
                if ep == p and ec == c:
                    edge_idx = i
                    break
            
            if edge_idx is None:
                # Should not happen
                continue
                
            ja = joint_angles[edge_idx]
            
            # Vector from p to c
            # In parent's local frame, angle is ja.
            # Global angle = ang[p] + ja
            global_angle = ang[p] + ja
            dist = 2 * np.sqrt(3)
            
            pos[c] = pos[p] + dist * np.array([np.cos(global_angle), np.sin(global_angle)])
            ang[c] = rot_offsets[c]
            
            queue.append(c)
            
    return pos, ang

def _objective(params):
    pos, ang = _compute_positions(params)
    
    # Minimize outer side
    side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    
    # Penalty for non-contact overlaps (SAT)
    # We only check pairs that are NOT in CONTACTS but might overlap.
    # Actually, we should check ALL pairs for overlap.
    penalty = 0.0
    
    verts = [_hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap_fast(verts[i], verts[j]):
                # Estimate overlap magnitude via center distance vs min dist
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = 2 * np.sqrt(3)
                if d < min_d:
                    penalty += 100.0 * (min_d - d)**2
                else:
                    # Overlapping despite center distance? (due to rotation)
                    penalty += 500.0
    
    # Penalty for breaking intended contacts (soft constraint to maintain topology)
    target_dist = 2 * np.sqrt(3)
    contact_penalty = 0.0
    for i, j in CONTACTS:
        d = np.linalg.norm(pos[i] - pos[j])
        contact_penalty += (d - target_dist)**2
        
    return side + 10.0 * contact_penalty + penalty

def construct():
    global START_TIME, DEADLINE
    
    # Initial parameters
    # Joint angles for the tree edges
    # 0->1: 0 deg
    # 1->2: 60 deg (relative to 0->1 vector? No, relative to 1's local frame)
    # Let's assume rot_offsets are 0 initially.
    # ang[0] = 0
    # Edge 0->1: global angle 0. pos[1] = (2sqrt3, 0).
    # Edge 1->2: We want 2 to be at 60 deg from center.
    # Vector 0->1 is along x-axis.
    # At 1, we want vector 1->2 to go such that 2 is at (sqrt3, 3).
    # pos[2] - pos[1] = (sqrt3 - 2sqrt3, 3 - 0) = (-sqrt3, 3).
    # Angle of this vector: atan2(3, -sqrt3) = 120 deg.
    # Since ang[1]=0, joint_angle = 120 deg.
    
    # This is getting complex to initialize perfectly.
    # Let's just use a random initialization close to the known solution.
    # Known solution positions:
    # 0: (0,0)
    # 1: (2sqrt3, 0)
    # 2: (sqrt3, 3)
    # 3: (-sqrt3, 3)
    # 4: (-2sqrt3, 0)
    # 5: (-sqrt3, -3)
    # 6: (sqrt3, -3)
    # 7: (3sqrt3, 3)  # approx
    # ...
    
    # Instead, let's use the initial configuration from the seed program as a guide for positions,
    # and solve for joint angles? No, just random restarts.
    
    best_pos = None
    best_ang = None
    best_side = float('inf')
    
    # Fixed seed for reproducibility if needed, but random restarts are better
    np.random.seed(42)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > DEADLINE - 20.0:
            break
            
        # Random joint angles
        joint_angles = np.random.uniform(-np.pi/6, np.pi/6, N-1)
        # Add some structure: start with regular hexagon ring angles
        if r == 0:
            # Approximate regular configuration
            joint_angles = np.array([0, 120, 120, 120, 120, 120, 180, 180, 180, 180]) * np.pi / 180
            # This might not be consistent. Let's just perturb the regular ones.
            # Regular ring:
            # 0->1: 0
            # 1->2: 120
            # 2->3: 120
            # 3->4: 120
            # 4->5: 120
            # 5->6: 120
            # 6->? 6 is leaf in my tree? No, 6 is child of 5.
            # My tree: 0-1-2-3-4-5-6.
            # So 6 is leaf.
            # 7 child of 2.
            # 8 child of 4.
            # 9 child of 5.
            # 10 child of 1.
            
            # Let's try to set joint angles to match the regular hexagon cluster.
            # pos[1] = (2sqrt3, 0). ang[1]=0.
            # pos[2] = (sqrt3, 3). Vector 1->2 is (-sqrt3, 3). Angle 120. ja[1] = 120 deg.
            # pos[3] = (-sqrt3, 3). Vector 2->3 is (-2sqrt3, 0). Angle 180. ja[2] = 180 deg (since ang[2]=0).
            # pos[4] = (-2sqrt3, 0). Vector 3->4 is (-sqrt3, -3). Angle 240 (-120). ja[3] = 240 deg.
            # pos[5] = (-sqrt3, -3). Vector 4->5 is (sqrt3, -3). Angle 300 (-60). ja[4] = 300 deg.
            # pos[6] = (sqrt3, -3). Vector 5->6 is (2sqrt3, 0). Angle 0. ja[5] = 0 deg.
            
            # Outer:
            # 7 (child of 2) at (3sqrt3, 3)? No, 7 is at 30 deg, dist 6.
            # pos[7] = (6cos30, 6sin30) = (3sqrt3, 3).
            # pos[2] = (sqrt3, 3).
            # Vector 2->7 is (2sqrt3, 0). Angle 0. ja[6] = 0 deg.
            
            # 8 (child of 4) at (3sqrt3, -3)? No, 8 is at 150 deg?
            # Ring 2 angles: 30, 150, 210, 330.
            # 7 at 30. 8 at 150. 9 at 210. 10 at 330.
            # pos[8] = (6cos150, 6sin150) = (-3sqrt3, 3).
            # pos[4] = (-2sqrt3, 0).
            # Vector 4->8 is (-sqrt3, 3). Angle 120. ja[7] = 120 deg.
            
            # 9 (child of 5) at 210.
            # pos[9] = (6cos210, 6sin210) = (-3sqrt3, -3).
            # pos[5] = (-sqrt3, -3).
            # Vector 5->9 is (-2sqrt3, 0). Angle 180. ja[8] = 180 deg.
            
            # 10 (child of 1) at 330.
            # pos[10] = (6cos330, 6sin330) = (3sqrt3, -3).
            # pos[1] = (2sqrt3, 0).
            # Vector 1->10 is (sqrt3, -3). Angle -60 (300). ja[9] = 300 deg.
            
            joint_angles = np.array([0, 120, 180, 240, 300, 0, 0, 120, 180, 300]) * np.pi / 180
            
        rot_offsets = np.zeros(N)
        
        x0 = np.concatenate([joint_angles, rot_offsets])
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            pos_opt, ang_opt = _compute_positions(res.x)
            
            # Check validity
            side_opt = _min_outer_side(pos_opt, ang_opt, (0.0, 0.0), 0.0)
            
            # SAT check for all pairs
            valid = True
            verts_opt = [_hex_vertices(pos_opt[i,0], pos_opt[i,1], 1.0, ang_opt[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    if _check_overlap_fast(verts_opt[i], verts_opt[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and side_opt < best_side:
                best_side = side_opt
                best_pos = pos_opt.copy()
                best_ang = ang_opt.copy()
                
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        # Use simple initial config
        sqrt3 = np.sqrt(3)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            a = i * np.pi / 3
            pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
        for i in range(4):
            a = ring2_angles[i]
            pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        best_pos = pos
        best_ang = ang
        best_side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)

    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()

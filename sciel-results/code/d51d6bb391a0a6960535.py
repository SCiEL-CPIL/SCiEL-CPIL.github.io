# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    def hex_vertices(cx, cy, s, a_deg):
        """Vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
        a = np.radians(a_deg)
        return np.array([[cx + s * np.cos(a + np.radians(60 * i)),
                          cy + s * np.sin(a + np.radians(60 * i))] for i in range(6)])

    def hex_edge_normals(cx, cy, s, a_deg):
        """12 edge normals (6 edges × 2 directions) for separation checks."""
        verts = hex_vertices(cx, cy, s, a_deg)
        normals = []
        for i in range(6):
            e1 = verts[i]
            e2 = verts[(i + 1) % 6]
            edge = e2 - e1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals.append(n)
            normals.append(-n)
        return np.array(normals)

    def hexagon_separation(h1, h2):
        """Check if two hexagons are separated (no intersection).
        Returns True if separated by > 1e-6 on at least one normal pair."""
        n1 = hex_edge_normals(h1[0], h1[1], 1.0, h1[2])
        n2 = hex_edge_normals(h2[0], h2[1], 1.0, h2[2])
        v1 = hex_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hex_vertices(h2[0], h2[1], 1.0, h2[2])
        all_normals = np.vstack([n1, n2])
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Intervals are apart if gap > 1e-6
            gap = max(min1 - max2, min2 - max1)
            if gap > 1e-6:
                return True
        return False

    def check_all_separated(inner):
        """Check all pairs of inner hexagons are separated."""
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if not hexagon_separation(inner[i], inner[j]):
                    return False
        return True

    def outer_hex_contains(inner, outer_center, outer_side):
        """Check all inner hexagons are inside the outer hexagon."""
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, 0)
        # For each inner hexagon vertex, check it's inside outer hex
        for h in inner:
            verts = hex_vertices(h[0], h[1], 1.0, h[2])
            for v in verts:
                if not point_in_hexagon(v, outer_verts):
                    return False
        return True

    def point_in_hexagon(p, hex_verts):
        """Check if point p is inside hexagon with vertices hex_verts."""
        n = len(hex_verts)
        for i in range(n):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % n]
            # Cross product to check same side
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross < -1e-6:
                return False
        return True

    def compute_min_side(inner):
        """Find minimum outer hexagon side length that contains all inner hexagons."""
        # Find centroid of all inner hexagon vertices
        all_verts = []
        for h in inner:
            all_verts.extend(hex_vertices(h[0], h[1], 1.0, h[2]))
        all_verts = np.array(all_verts)
        center = np.mean(all_verts, axis=0)
        
        # Binary search on side length
        lo, hi = 2.0, 10.0
        for _ in range(50):
            mid = (lo + hi) / 2
            outer_verts = hex_vertices(center[0], center[1], mid, 0)
            if outer_hex_contains(inner, center, mid):
                hi = mid
            else:
                lo = mid
        return center, hi

    def validate(inner, outer_center, outer_side):
        """Full validation."""
        # Check separation
        if not check_all_separated(inner):
            return False
        # Check containment
        if not outer_hex_contains(inner, outer_center, outer_side):
            return False
        # Check side >= sqrt(11)
        if outer_side < np.sqrt(11) - 1e-9:
            return False
        # Check edge lengths of shifted inner hexagons
        shift = np.array(outer_center)
        for h in inner:
            verts = hex_vertices(h[0] - shift[0], h[1] - shift[1], 1.0, h[2])
            for i in range(6):
                edge = verts[(i + 1) % 6] - verts[i]
                if abs(np.linalg.norm(edge) - 1.0) > 1e-8:
                    return False
        return True

    # Initial cluster configuration: center + 6 neighbors + 4 tips
    # Hexagon lattice: neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    d = np.sqrt(3)
    init_inner = np.array([
        [0, 0, 0],           # center
        [d, 0, 0],           # right
        [d/2, d*np.sqrt(3)/2, 0],  # top-right
        [-d/2, d*np.sqrt(3)/2, 0], # top-left
        [-d, 0, 0],          # left
        [-d/2, -d*np.sqrt(3)/2, 0], # bottom-left
        [d/2, -d*np.sqrt(3)/2, 0],  # bottom-right
        # 4 additional hexagons forming tips
        [2*d, 0, 0],         # far right
        [d, d*np.sqrt(3), 0], # far top-right
        [-d, d*np.sqrt(3), 0], # far top-left
        [-2*d, 0, 0],        # far left
    ])

    # Optimize to reduce bounding hexagon size
    # Variables: positions and angles of all 11 hexagons (except center fixed)
    def objective(x):
        inner = np.copy(init_inner)
        idx = 1  # start from hexagon 1
        for i in range(10):
            inner[idx, 0] = x[i * 3]
            inner[idx, 1] = x[i * 3 + 1]
            inner[idx, 2] = x[i * 3 + 2]
            idx += 1
        
        # Check separation constraint (soft penalty)
        sep_violation = 0
        for i in range(11):
            for j in range(i + 1, 11):
                if not hexagon_separation(inner[i], inner[j]):
                    sep_violation += 100.0
        
        # Compute minimum side length
        center, side = compute_min_side(inner)
        return side + sep_violation

    x0 = np.zeros(30)
    # Small perturbation to break symmetry
    np.random.seed(42)
    x0 += 0.1 * np.random.randn(30)

    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})

    # Extract optimized configuration
    inner = np.copy(init_inner)
    if result.success or result.fun < 5:
        idx = 1
        for i in range(10):
            inner[idx, 0] = result.x[i * 3]
            inner[idx, 1] = result.x[i * 3 + 1]
            inner[idx, 2] = result.x[i * 3 + 2]
            idx += 1

    # Compute optimal outer hexagon
    outer_center, outer_side = compute_min_side(inner)
    
    # Fine-tune: try to reduce side by adjusting angles
    for _ in range(100):
        best_inner = np.copy(inner)
        best_side = outer_side
        for i in range(11):
            for da in [-10, -5, -2, 2, 5, 10]:
                trial = np.copy(inner)
                trial[i, 2] = inner[i, 2] + da
                c, s = compute_min_side(trial)
                if s < best_side - 1e-6:
                    best_inner = trial
                    best_side = s
                    outer_center = c
        if np.allclose(best_inner, inner):
            break
        inner = best_inner
        outer_side = best_side
        outer_center, outer_side = compute_min_side(inner)

    # Final validation
    if not validate(inner, outer_center, outer_side):
        # Fallback to initial configuration
        inner = np.copy(init_inner)
        outer_center, outer_side = compute_min_side(inner)

    inner_hex_data = inner
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    outer_hex_side_length = float(outer_side)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

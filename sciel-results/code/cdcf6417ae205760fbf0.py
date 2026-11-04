import numpy as np
import time

N = 11

def hexagon_support(theta):
    # Support function of unit hexagon (side 1, vertex at 0)
    # theta in radians
    # h(theta) = max(cos(theta - k*pi/3))
    k = np.arange(6)
    return np.max(np.cos(theta - k * np.pi / 3))

def cluster_support(centers, theta):
    # centers: Nx2 array
    # theta: scalar
    u = np.array([np.cos(theta), np.sin(theta)])
    proj = np.max(centers @ u)
    return hexagon_support(theta) + proj

def container_support(S, theta):
    # Support function of regular hexagon side S
    # Apothem = S * sqrt(3) / 2
    # H(theta) = Apothem / cos(theta - 30 deg) ?
    # Check: theta=0 (vertex). cos(-30) = sqrt(3)/2. H = (S*sqrt(3)/2) / (sqrt(3)/2) = S. Correct.
    # theta=30 (side normal). cos(0) = 1. H = S*sqrt(3)/2. Correct.
    return (S * np.sqrt(3) / 2) / np.cos(theta - np.pi / 6)

def min_enclosing_hexagon_side(centers):
    # Binary search for S
    lo, hi = 0.0, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        # Check if fits
        thetas = np.linspace(0, 2 * np.pi, 100)
        h_c = np.array([cluster_support(centers, t) for t in thetas])
        h_cont = np.array([container_support(mid, t) for t in thetas])
        if np.all(h_c <= h_cont):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    
    # 7-cluster centers
    r = np.sqrt(3)
    angles = np.arange(6) * np.pi / 3
    centers_7 = np.array([[0,0]])
    for a in angles:
        centers_7 = np.vstack([centers_7, [r * np.cos(a), r * np.sin(a)]])
    
    # Candidate positions for 4 extra hexagons
    # Lattice points at distance 2r from origin, or adjacent to the 7-cluster
    # We will search for 4 positions that minimize the side
    
    # Generate candidates: all lattice points within distance 2.5 * r from origin
    # Lattice basis: v1 = (r, 0), v2 = (r/2, 3r/2)? No.
    # Hex lattice basis:
    # e1 = (r, 0)
    # e2 = (r * np.cos(np.pi/3), r * np.sin(np.pi/3)) = (r/2, r*sqrt(3)/2)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * np.array([r, 0]) + j * np.array([r/2, r*sqrt(3)/2])
            # Check if p is not in centers_7
            # And distance from origin < 3*r
            if np.linalg.norm(p) < 3 * r:
                # Check overlap with existing 7
                overlap = False
                for c in centers_7:
                    if np.linalg.norm(p - c) < r - 1e-6:
                        overlap = True
                        break
                if not overlap:
                    candidates.append(p)
    
    candidates = np.array(candidates)
    
    best_side = 100.0
    best_extra = None
    
    # Greedy selection of 4 extra centers
    current_centers = centers_7.copy()
    used = [False] * len(candidates)
    
    for _ in range(4):
        best_candidate_idx = -1
        best_candidate_side = 100.0
        for idx, p in enumerate(candidates):
            if used[idx]:
                continue
            # Check overlap with current_centers
            overlap = False
            for c in current_centers:
                if np.linalg.norm(p - c) < r - 1e-6:
                    overlap = True
                    break
            if overlap:
                continue
            
            new_centers = np.vstack([current_centers, [p]])
            # Estimate side (use a rough check for speed)
            # Just compute min_enclosing_hexagon_side
            side = min_enclosing_hexagon_side(new_centers)
            if side < best_candidate_side:
                best_candidate_side = side
                best_candidate_idx = idx
        
        if best_candidate_idx != -1:
            current_centers = np.vstack([current_centers, [candidates[best_candidate_idx]]])
            used[best_candidate_idx] = True
            best_side = best_candidate_side
            
        if time.time() - start_time > 140:
            break

    # Final computation
    final_side = min_enclosing_hexagon_side(current_centers)
    
    # Convert centers to (x, y, angle)
    inner = []
    for c in current_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END

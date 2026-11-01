import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def compute_radius(pts):
        n = len(pts)
        if n < 2:
            return np.inf
        # Pairwise minimum distance / 2
        min_pair = np.inf
        for i in range(n):
            diffs = pts[i+1:] - pts[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if dists.size > 0:
                min_pair = min(min_pair, np.min(dists))
        min_pair /= 2.0
        # Boundary minimum
        radii = np.sqrt(np.sum(pts**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        return min(min_pair, min_boundary)

    # --- Step 1: Concentric ring initialization (matches known optimal structure) ---
    # The best known packing for 300 circles uses concentric rings.
    # Ring 0: 1 circle at center
    # Ring k: k * m circles (where m is chosen per ring)
    # We'll try several ring configurations.

    def ring_init(n, ring_counts):
        """Place n points on concentric rings. ring_counts is a list of counts per ring."""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        radius = 0.0
        for k, count in enumerate(ring_counts):
            if count == 0:
                continue
            # Radius for this ring: roughly proportional to sqrt(k+1) to equalize density
            # Better: use the fact that area per ring ~ 2*pi*r*dr, so r ~ sqrt(k)
            radius = 0.15 * (k + 1)  # Will be adjusted by relaxation
            # Actually, let's space rings so that the gap between rings ~ 2r
            # Total area = pi, N circles each area pi*r^2, so N*r^2 <= 1, r <= 1/sqrt(N)
            # For N=300, r_max ~ 0.0577. Ring spacing should be ~ 2*r ~ 0.11
            # Let's use r_ring_k = (k+1) * ring_spacing
            pass

        # Better approach: compute ring radii to maximize min distance
        # For a ring of count c at radius R, the angular spacing gives chord = 2*R*sin(pi/c)
        # The inter-ring gap is R_{k+1} - R_k
        # We want: 2*R_k*sin(pi/c_k) >= 2*r and R_{k+1} - R_k >= 2*r
        # And R_max + r <= 1

        # Let's just use evenly spaced rings and let relaxation fix it
        # Number of rings
        n_rings = len(ring_counts)
        # Distribute radii from 0 to ~0.95
        ring_radii = np.linspace(0.1, 0.95, n_rings) if n_rings > 1 else np.array([0.0])

        pts = []
        for k, count in enumerate(ring_counts):
            if count <= 0:
                continue
            if k == 0 and count == 1:
                pts.append([0.0, 0.0])
                continue
            R = ring_radii[k] if k < len(ring_radii) else 0.5
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                pts.append([R * np.cos(a), R * np.sin(a)])

        return np.array(pts[:n])

    # Try a few ring configurations
    # Known structure for ~300 circles: roughly 1, 6, 12, 18, 24, 30, 36, 42, ...
    # Sum: 1+6+12+18+24+30+36+42 = 169, need 131 more
    # 1+6+12+18+24+30+36+42+48 = 217, need 83 more
    # 1+6+12+18+24+30+36+42+48+54 = 271, need 29 more
    # 1+6+12+18+24+30+36+42+48+54+29 = 300

    configs = [
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],  # 300
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 63],      # 280 -> adjust
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 30],  # 301 -> trim
        [1, 8, 16, 24, 32, 40, 48, 56, 48, 27],      # 300
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],  # same as first
    ]

    best_pts = None
    best_radius = 0.0

    for cfg in configs:
        total = sum(cfg)
        if total < N:
            # Add remainder to last ring
            cfg = cfg + [N - total]
        elif total > N:
            # Trim from last ring
            excess = total - N
            cfg[-1] -= excess
            if cfg[-1] <= 0:
                cfg.pop()
                # Redistribute
                while sum(cfg) < N and len(cfg) > 1:
                    cfg[-1] += 1
                if sum(cfg) < N:
                    cfg.append(N - sum(cfg))
                elif sum(cfg) > N:
                    cfg[-1] -= sum(cfg) - N
                    if cfg[-1] <= 0:
                        cfg.pop()

        if sum(cfg) != N:
            continue

        pts = ring_init(N, cfg)
        if len(pts) < N:
            continue

        r_init = compute_radius(pts)
        if best_pts is None or r_init > best_radius:
            best_radius = r_init
            best_pts = pts.copy()

    # Also try hexagonal lattice as fallback
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.98

    pts_hex = hex_lattice_init(N)
    r_hex = compute_radius(pts_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_pts = pts_hex.copy()

    # --- Step 2: Force-directed relaxation ---
    # Each point feels:
    # 1. Repulsion from all other points (like charges) - pushes apart
    # 2. Attraction to center (weak) - keeps them in
    # 3. Hard constraint: stay inside unit circle
    # 4. We want to maximize the minimum distance, so we use a potential that
    #    penalizes close pairs strongly

    def force_directed_relax(pts, max_iter=500, dt_init=0.005):
        """Iterative force-directed relaxation to maximize min distance."""
        n = len(pts)
        pts = pts.copy()
        dt = dt_init

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Compute forces
            forces = np.zeros_like(pts)

            # Pairwise repulsion: F_ij = k * (c_i - c_j) / d_ij^3 for d_ij < threshold
            # Use a soft repulsion that's strong at close range
            # We want to maximize min distance, so use:
            # F_i = sum_{j != i} (c_i - c_j) / max(d_ij, eps)^2 * strength
            # This is like Coulomb repulsion

            # For efficiency with 300 points, O(n^2) = 90000 ops per iteration is fine
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    d = np.sqrt(np.dot(diff, diff))
                    if d < 1e-12:
                        # Random nudge
                        forces[i] += np.random.randn(2) * 0.01
                        forces[j] -= np.random.randn(2) * 0.01
                        continue
                    # Repulsion strength: stronger when close
                    # Use 1/d^2 for Coulomb-like, but cap it
                    strength = 1.0 / (d * d + 1e-6)
                    f = diff / d * strength
                    forces[i] += f
                    forces[j] -= f

            # Weak centering force (to prevent drift)
            for i in range(n):
                forces[i] -= 0.01 * pts[i]

            # Update positions
            pts += dt * forces

            # Enforce boundary: project back inside unit circle
            radii = np.sqrt(np.sum(pts**2, axis=1))
            for i in range(n):
                if radii[i] > 0.999:
                    pts[i] = pts[i] / radii[i] * 0.999

            # Adaptive step size: reduce if any pair is too close
            min_d = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                    if d < min_d:
                        min_d = d

            # If min distance decreased, reduce step
            if min_d < 0.01:
                dt *= 0.9
            else:
                dt = min(dt * 1.01, 0.005)

            # Check convergence
            max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
            if max_force < 1e-8:
                break

        return pts

    # Run force-directed relaxation
    pts_relaxed = force_directed_relax(best_pts, max_iter=300, dt_init=0.003)
    r_relaxed = compute_radius(pts_relaxed)

    if r_relaxed > best_radius:
        best_radius = r_relaxed
        best_pts = pts_relaxed.copy()

    # --- Step 3: Local search - push the closest pair apart ---
    # Identify the constraining pair and nudge them apart
    if time.time() < deadline - 10:
        for trial in range(200):
            if time.time() > deadline - 3:
                break

            pts = best_pts.copy()
            n = len(pts)

            # Find closest pair
            min_d = np.inf
            min_i, min_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                    if d < min_d:
                        min_d = d
                        min_i, min_j = i, j

            # Nudge the closest pair apart
            diff = pts[min_i] - pts[min_j]
            d = np.sqrt(np.sum(diff**2))
            if d < 1e-10:
                diff = np.random.randn(2)
                d = np.sqrt(np.sum(diff**2))

            # Push apart by a small amount
            push = 0.002
            pts[min_i] += diff / d * push
            pts[min_j] -= diff / d * push

            # Also nudge all points slightly
            for i in range(n):
                # Push away from center if too close to boundary
                r_i = np.sqrt(np.sum(pts[i]**2))
                if r_i > 0.98:
                    pts[i] = pts[i] / r_i * 0.98

                # Small random perturbation
                pts[i] += np.random.randn(2) * 0.0005

                # Keep inside
                r_new = np.sqrt(np.sum(pts[i]**2))
                if r_new > 0.999:
                    pts[i] = pts[i] / r_new * 0.999

            r_new = compute_radius(pts)
            if r_new > best_radius:
                best_radius = r_new
                best_pts = pts.copy()

    # --- Step 4: Final force-directed pass with smaller steps ---
    if time.time() < deadline - 5:
        pts_final = force_directed_relax(best_pts, max_iter=100, dt_init=0.001)
        r_final = compute_radius(pts_final)
        if r_final > best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()

    # Final validation
    r_final_check = compute_radius(best_pts)
    if r_final_check > best_radius:
        best_radius = r_final_check

    # Ensure all points are strictly inside
    radii = np.sqrt(np.sum(best_pts**2, axis=1))
    for i in range(len(best_pts)):
        if radii[i] >= 1.0:
            best_pts[i] = best_pts[i] / radii[i] * 0.9999

    return best_pts

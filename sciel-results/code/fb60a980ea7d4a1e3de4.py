"""Sutton-Chen cluster global-minima search: diverse structural prototypes + L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 330s limit


def _remaining():
    """Return seconds remaining before the deadline."""
    return _DEADLINE_MARGIN + (330.0 - (time.time() - _START_TIME))


def _valid(pos):
    """Check if positions are finite and no two atoms closer than 0.35."""
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _generate_fcc_cluster(n):
    """Generate n atoms from an FCC lattice, centered."""
    pts = []
    # FCC basis
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    # Generate from a cubic grid
    max_coord = 10
    all_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    all_pts.append(p)
    all_pts = np.array(all_pts)
    # Center
    centroid = all_pts.mean(axis=0)
    all_pts = all_pts - centroid
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = all_pts[idx]
    # Scale to reasonable size
    if n > 1:
        # Target average nearest-neighbor distance ~1.0
        # For FCC, nearest neighbor distance is a/sqrt(2) where a is lattice constant
        # We want the cluster to have a reasonable extent
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            target_radius = (n / (4.0 / 3.0 * np.pi * 1.0**3)) ** (1.0 / 3.0) * 1.2
            pts = pts * (target_radius / max_r)
    return pts


def _generate_bcc_cluster(n):
    """Generate n atoms from a BCC lattice, centered."""
    pts = []
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    max_coord = 10
    all_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    all_pts.append(p)
    all_pts = np.array(all_pts)
    centroid = all_pts.mean(axis=0)
    all_pts = all_pts - centroid
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = all_pts[idx]
    if n > 1:
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            target_radius = (n / (4.0 / 3.0 * np.pi * 1.0**3)) ** (1.0 / 3.0) * 1.2
            pts = pts * (target_radius / max_r)
    return pts


def _generate_hcp_cluster(n):
    """Generate n atoms from an HCP lattice, centered."""
    # HCP: close-packed layers in ABAB stacking
    a = 1.0  # in-plane lattice constant
    c = np.sqrt(8.0 / 3.0) * a  # c/a ratio for ideal HCP
    pts = []
    # Generate layers
    max_layer = 15
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        # Each layer is a triangular lattice
        # For ABAB: even layers have one orientation, odd layers shifted
        offset = 0.0 if layer % 2 == 0 else a * np.array([1.0 / 3.0, 2.0 / 3.0, 0.0])
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                x = i * a + offset[0]
                y = j * a * np.sqrt(3.0) / 2.0 + offset[1] + (i % 2) * a * np.sqrt(3.0) / 4.0
                pts.append([x, y, z])
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx]
    if n > 1:
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            target_radius = (n / (4.0 / 3.0 * np.pi * 1.0**3)) ** (1.0 / 3.0) * 1.2
            pts = pts * (target_radius / max_r)
    return pts


def _generate_icosahedron(n):
    """Generate atoms in an icosahedral arrangement."""
    # Start with a Mackay icosahedron or similar
    # For small n, use a single icosahedron (12 vertices + 1 center = 13)
    # For larger n, use nested icosahedra or icosahedral shells
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])  # Normalize
    
    pts = []
    # Center atom
    pts.append([0, 0, 0])
    
    if n <= 13:
        # Just use the icosahedron
        ico = ico_verts[:min(n - 1, 12)]
        pts.extend(ico.tolist())
        pts = np.array(pts[:n])
    else:
        # Add shells
        remaining = n - 1
        # First shell: 12 icosahedron vertices
        shell1 = ico_verts.copy()
        pts.extend(shell1.tolist())
        remaining -= 12
        
        if remaining > 0:
            # Second shell: scale up
            scale2 = 2.0
            shell2 = ico_verts * scale2
            # Add more points on the second shell
            # Use icosahedron vertices and midpoints of edges
            ico_mids = []
            for i in range(12):
                for j in range(i + 1, 12):
                    # Check if they're neighbors (distance ~2*sqrt(2))
                    d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    if d < 2.5:  # Approximate neighbor check
                        mid = (ico_verts[i] + ico_verts[j]) / 2.0
                        mid = mid / np.linalg.norm(mid) * scale2
                        ico_mids.append(mid)
            # Remove duplicates
            unique_mids = []
            for m in ico_mids:
                if not any(np.linalg.norm(m - u) < 0.1 for u in unique_mids):
                    unique_mids.append(m)
            shell2_extra = np.array(unique_mids[:remaining])
            pts.extend(shell2_extra.tolist())
            remaining -= len(shell2_extra)
        
        if remaining > 0:
            # Third shell
            scale3 = 3.0
            shell3 = ico_verts * scale3
            pts.extend(shell3.tolist()[:remaining])
    
    pts = np.array(pts[:n])
    if n > 1:
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            target_radius = (n / (4.0 / 3.0 * np.pi * 1.0**3)) ** (1.0 / 3.0) * 1.2
            pts = pts * (target_radius / max_r)
    return pts


def _generate_random_compact(n, rng, spacing=0.72):
    """Generate a random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def construct(instance):
    """Search for global minimum of Sutton-Chen potential cluster."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    rng = np.random.default_rng(size + 42)
    
    generators = [
        ("fcc", _generate_fcc_cluster),
        ("bcc", _generate_bcc_cluster),
        ("hcp", _generate_hcp_cluster),
        ("ico", _generate_icosahedron),
    ]
    
    # Try each generator
    for name, gen in generators:
        if _remaining() < 5.0:
            break
        try:
            pts = gen(size)
            if _valid(pts):
                energy, positions = relax(pts, potential, kind)
                if _valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
    
    # Try several random compact structures
    n_random = 6
    for i in range(n_random):
        if _remaining() < 5.0:
            break
        try:
            pts = _generate_random_compact(size, rng)
            if _valid(pts):
                energy, positions = relax(pts, potential, kind)
                if _valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
    
    # If no valid result, return a simple compact structure
    if best_positions is None:
        pts = _generate_random_compact(size, rng)
        if not _valid(pts):
            # Fallback: linear chain
            pts = np.zeros((size, 3))
            pts[:, 0] = np.arange(size) * 1.0
        best_positions = pts
    
    return best_positions
# EVOLVE-BLOCK-END

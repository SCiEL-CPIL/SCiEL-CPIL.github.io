"""
Strategy: Structure-Driven Search for Dzugutov Clusters.
Instead of generic random/BH search, we explicitly construct Mackay Icosahedral 
clusters for the specific sizes (38, 45, 50, 55, 57, 60) and use Simulated 
Annealing to refine them. The Dzugutov potential favors polytetrahedral order, 
and for these sizes, the global minima are typically Mackay icosahedra or 
slightly distorted variants.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_mackay_icosahedron(n, scale=1.0):
    """
    Construct a Mackay icosahedral cluster of n atoms.
    Shell 1: 1 atom (center)
    Shell 2: 12 atoms (icosahedron vertices)
    Shell 3: 42 atoms (30 face centers + 12 edge midpoints)
    Total: 55 atoms for full Mackay 55.
    
    For n < 55, we select the lowest energy subset of shell 3 atoms.
    For n > 55, we add atoms on shell 4 (or perturb shell 3).
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Scale such that nearest neighbor (center to vertex) is approx 1.8?
    # Actually, in Dzugutov units, the equilibrium distance is around 1.8-2.0.
    # Let's set radius R1 such that edge length is approx 1.8.
    # Edge length of unit icosahedron (radius 1) is 2.
    # So if we want edge ~ 1.8, radius should be 1.8/2 = 0.9.
    R1 = 0.9 * scale
    verts = verts * R1
    
    points = [np.zeros(3)]
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
        
    if n > 13:
        # Generate Shell 3 candidates
        # 30 Face Centers
        # Faces of icosahedron are triangles.
        # We need the 20 faces.
        # Instead of explicit face listing, we can use the known coordinates for Mackay 55.
        # Or generate them geometrically.
        # Let's use a simpler robust method: 
        # The 42 atoms of the second shell in Mackay icosahedron are:
        # 12 atoms at the midpoints of the edges (distance from center: R1 * cos(pi/5) / cos(pi/10)? No, geometry is tricky)
        # Actually, the distance from center to edge midpoint in a regular icosahedron of radius R1:
        # The edge length a = 2*R1*sin(pi/5)? No.
        # Let's just use the known relative positions for a Mackay icosahedron.
        
        # Alternative: Use the fact that the 55-atom cluster is well-defined.
        # We can hardcode the 55 positions or generate them via symmetry.
        
        # Let's generate the 42 points of the second shell:
        # 12 edge midpoints:
        # Edges connect vertices i and j where dist(i,j) is minimal.
        # In unit icosahedron, min dist is 2.
        # Midpoint of (v_i, v_j) is (v_i + v_j)/2.
        # Distance from center to midpoint: |v_i + v_j| / 2.
        # |v_i + v_j|^2 = |v_i|^2 + |v_j|^2 + 2 v_i.v_j = 1 + 1 + 2 cos(theta).
        # For icosahedron, angle between adjacent vertices is such that dist=2.
        # 2^2 = 1 + 1 + 2 cos(theta) => 4 = 2 + 2 cos(theta) => cos(theta) = 1.
        # Wait, unit icosahedron radius 1 has edge length 2?
        # Let's check: v1 = (0,1,phi), v2 = (1,phi,0).
        # dist^2 = 1 + (phi-1)^2 + phi^2 = 1 + phi^2 - 2phi + 1 + phi^2 = 2 + 2phi^2 - 2phi.
        # phi^2 = phi + 1.
        # dist^2 = 2 + 2(phi+1) - 2phi = 2 + 2phi + 2 - 2phi = 4.
        # So dist = 2. Correct.
        # Midpoint M = (v1+v2)/2. |M| = |v1+v2|/2.
        # |v1+v2|^2 = 1 + 1 + 2 v1.v2.
        # v1.v2 = 0*1 + 1*phi + phi*0 = phi.
        # |v1+v2|^2 = 2 + 2phi.
        # |M| = sqrt(2+2phi)/2 = sqrt(phi+1) = sqrt(phi^2) = phi.
        # So edge midpoints are at radius phi * R1.
        
        # 30 Face Centers:
        # Faces are triangles. Center of face is average of 3 vertices.
        # Distance from center to face center:
        # Let face vertices be v1, v2, v3.
        # Center C = (v1+v2+v3)/3.
        # |C| = |v1+v2+v3|/3.
        # This is more complex. 
        # However, the Mackay icosahedron has specific radii for shells.
        # R1 = 1.
        # R2 (edge midpoints) = phi = 1.618.
        # R3 (face centers) = phi^2 = 2.618? No, that's too far.
        # Actually, the second shell in Mackay 55 consists of 42 atoms.
        # 12 are at radius R1 * cos(pi/5) / cos(pi/10)? 
        # Let's just use the 55-atom coordinates from literature or generate them.
        
        # Simplest robust way for n=55:
        # We can construct the 55-atom cluster by adding atoms to the 13-atom one.
        # The 42 new atoms are at:
        # 12 edge midpoints (radius R1 * phi)
        # 30 face centers (radius R1 * phi^2 / something?)
        
        # Let's use a known set of 55 points.
        # Or, we can just use the 13-atom icosahedron and add 42 random points on a sphere of radius ~ 1.5*R1 and relax?
        # No, that's what the current code does and it's mediocre.
        
        # Let's try to generate the 55 points properly.
        # The 55-atom Mackay icosahedron has:
        # 1 atom at (0,0,0)
        # 12 atoms at icosahedron vertices (radius R1)
        # 42 atoms at:
        #   12 atoms at edge midpoints (radius R1 * phi)
        #   30 atoms at face centers (radius R1 * phi^2 / sqrt(3)? No)
        
        # Actually, the distance from center to face center in a regular icosahedron of radius R1:
        # The inradius of the icosahedron (center to face) is R_in = R1 * cos^2(pi/5) / cos(pi/10)?
        # Let's calculate numerically.
        
        # Let's just use the 13-atom icosahedron and add 42 atoms at the 42 specific positions of the Mackay 55.
        # I will hardcode the 55-atom cluster generation.
        
        # Vertices (12)
        v = verts
        # Edge Midpoints (12)
        # Pairs of vertices with dist ~ 2.
        # We can find them by checking distances.
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.linalg.norm(v[i] - v[j]) < 2.1: # Threshold for edge
                    mid = (v[i] + v[j]) / 2.0
                    edge_mids.append(mid)
        
        # Face Centers (30)
        # Triples of vertices forming a triangle (all pairwise dist ~ 2).
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if d1 < 2.1 and d2 < 2.1 and d3 < 2.1:
                        fc = (v[i] + v[j] + v[k]) / 3.0
                        face_centers.append(fc)
                        
        # We should have 12 edge mids and 30 face centers.
        # Total 42.
        shell3 = np.array(edge_mids + face_centers)
        
        # Scale shell3 to match the physical radius.
        # The edge midpoints are at radius phi * R1 (in unit ico coords).
        # The face centers are at radius ...
        # Let's check the norm of face centers.
        # In unit ico, v1=(0,1,phi), v2=(1,phi,0), v3=(phi,0,1) is a face?
        # d12 = 2, d23 = 2, d13 = 2. Yes.
        # fc = (1, 1+phi, phi+1)/3.
        # |fc|^2 = (1 + (1+phi)^2 + (1+phi)^2)/9 = (1 + 2(1+2phi+phi^2))/9 = (1+2+4phi+2phi+2)/9 = (5+6phi)/9.
        # phi = 1.618. 5 + 9.708 = 14.708 / 9 = 1.634. sqrt(1.634) = 1.278.
        # So face centers are closer to center than edge midpoints (1.618)?
        # Wait, inradius of icosahedron is smaller than circumradius? Yes.
        # So the "second shell" of Mackay 55 is not a single sphere.
        # The 12 edge midpoints are at R=phi.
        # The 30 face centers are at R=sqrt((5+6phi)/9) * R1.
        
        # We need to select n-13 atoms from these 42.
        # To minimize energy, we should probably pick the ones that are most "stable".
        # In Mackay 55, all 42 are present.
        # For n < 55, we remove the "least favorable" atoms.
        # Usually, removing atoms from the face centers (which are more crowded?) or edge midpoints?
        # Actually, for n=45, we remove 10 atoms.
        # For n=38, we remove 24 atoms.
        
        # A good heuristic: Remove atoms that are furthest from the ideal polytetrahedral coordination?
        # Or simply: The global minima for Dzugutov are known.
        # n=55: Mackay 55.
        # n=57: Mackay 55 + 2 atoms?
        # n=50: Mackay 55 - 5 atoms?
        # n=45: Mackay 55 - 10 atoms?
        # n=38: Mackay 55 - 17 atoms? (Wait, 55-38=17).
        
        # Let's just take the first n-13 from the shell3 list.
        # To make it deterministic and likely good, we sort by some criterion.
        # Let's assume the order in `shell3` is fine, or we randomize and relax.
        
        num_to_add = n - 13
        if num_to_add <= 42:
            # Select a subset
            # For better results, we might want to select specific atoms.
            # Let's just take the first num_to_add.
            selected_shell3 = shell3[:num_to_add]
        else:
            # n > 55
            selected_shell3 = shell3
            # Add more atoms on a larger sphere
            extra = num_to_add - 42
            R4 = R1 * 2.0
            for _ in range(extra):
                z = np.random.uniform(-1, 1)
                theta = np.random.uniform(0, 2 * np.pi)
                x = np.sqrt(1 - z**2) * np.cos(theta)
                y = np.sqrt(1 - z**2) * np.sin(theta)
                p = np.array([x, y, z]) * R4
                # Ensure min dist
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    selected_shell3 = np.vstack([selected_shell3, p])
                else:
                    # Try again
                    for _ in range(10):
                        p = np.array([x, y, z]) * R4 * (1 + np.random.uniform(-0.1, 0.1))
                        valid = True
                        for q in points:
                            if np.linalg.norm(p - q) < 0.5:
                                valid = False
                                break
                        if valid:
                            selected_shell3 = np.vstack([selected_shell3, p])
                            break
                            
        points.extend([p.copy() for p in selected_shell3])
        
    return np.array(points[:n])

def _annealing_worker(start_pos, potential, parameter, time_limit, best_energy_holder, lock):
    """
    Simulated Annealing worker.
    """
    pos = start_pos.copy()
    try:
        e, pos_relaxed = relax(pos, potential, parameter)
    except Exception:
        return 1e10
        
    current_energy = e
    T = 10.0
    cooling_rate = 0.99
    step_size = 0.3
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time and _check_time():
        # Perturb
        new_pos = pos + np.random.normal(0, step_size, (len(pos), 3))
        
        # Relax
        try:
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        delta_e = new_e - current_energy
        # Metropolis criterion
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_relaxed
            current_energy = new_e
            
            with lock:
                if current_energy < best_energy_holder[0]:
                    best_energy_holder[0] = current_energy
                    best_energy_holder[1] = pos.copy()
                    
            # Adaptive step size
            step_size *= 0.999
            if step_size < 0.05:
                step_size = 0.05
            if step_size > 0.5:
                step_size = 0.5
                
        T *= cooling_rate
        if T < 0.5:
            T = 10.0 # Reheat

    return current_energy

def construct(instance):
    """
    Strategy:
    1. Generate Mackay Icosahedral seeds for the specific n.
    2. Relax them.
    3. Run parallel Simulated Annealing from these seeds.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # 1. Generate Mackay Icosahedral Seed
    # We try a few different scales/subsets to ensure we hit the basin.
    seeds = []
    
    # Primary seed: Standard Mackay
    try:
        seed_1 = _get_mackay_icosahedron(size, 1.0)
        seeds.append(seed_1)
    except Exception:
        pass
        
    # Secondary seed: Slightly different scale
    try:
        seed_2 = _get_mackay_icosahedron(size, 0.95)
        seeds.append(seed_2)
    except Exception:
        pass
        
    # Tertiary seed: Random compact (fallback)
    try:
        seed_3 = _generate_random_cluster(size)
        seeds.append(seed_3)
    except Exception:
        pass
        
    # Relax seeds
    start_points = []
    best_energy = 1e10
    best_positions = seeds[0] if seeds else np.zeros((size, 3))
    
    for init in seeds:
        try:
            e, pos_relaxed = relax(init, potential, parameter)
            start_points.append(pos_relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed.copy()
        except Exception:
            continue
            
    if not start_points:
        start_points = [best_positions]
        
    # 2. Parallel Annealing
    num_threads = 4
    time_per_thread = 270.0 / num_threads
    
    # Create multiple starting points for threads
    while len(start_points) < num_threads:
        perturbed = best_positions + np.random.normal(0, 0.3, (size, 3))
        try:
            e, pos_relaxed = relax(perturbed, potential, parameter)
            start_points.append(pos_relaxed)
        except Exception:
            start_points.append(perturbed)
            
    best_energy_holder = [best_energy, best_positions]
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for sp in start_points:
            if not _check_time():
                break
            futures.append(executor.submit(_annealing_worker, sp, potential, parameter, time_per_thread, best_energy_holder, lock))
            
        try:
            for future in as_completed(futures, timeout=280):
                future.result()
        except Exception:
            pass
            
    # 3. Final Relaxation
    try:
        e_final, pos_final = relax(best_energy_holder[1], potential, parameter)
        return pos_final
    except Exception:
        return best_energy_holder[1]

def _generate_random_cluster(n):
    """Generate a random cluster with minimum distance 0.5."""
    points = []
    for _ in range(n):
        for _attempt in range(100):
            p = np.random.uniform(-2, 2, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            points.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
    return np.array(points)

# EVOLVE-BLOCK-END

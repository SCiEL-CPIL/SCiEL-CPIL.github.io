<<<<<<< SEARCH
def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_ratio_spiral(n=N, start_offset=0.0):
    """Generate points using golden ratio spacing with different starting offset."""
    i = np.arange(n) + start_offset
    z = 1.0 - 2.0 * (i + 0.5) / n
    golden = 2.39996322972865332  # 2*pi / phi^2
    phi_angle = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add perturbed Fibonacci points
    fib = _fibonacci_spiral(n)
    noise = np.random.normal(0, 0.05, size=(n, 3))
    pts = fib + noise
    return _project_to_sphere(pts)
=======
def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_ratio_spiral(n=N, start_offset=0.0):
    """Generate points using golden ratio spacing with different starting offset."""
    i = np.arange(n) + start_offset
    z = 1.0 - 2.0 * (i + 0.5) / n
    golden = 2.39996322972865332  # 2*pi / phi^2
    phi_angle = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add perturbed Fibonacci points
    fib = _fibonacci_spiral(n)
    noise = np.random.normal(0, 0.05, size=(n, 3))
    pts = fib + noise
    return _project_to_sphere(pts)

def _mesh_based_init(n=N):
    """
    Generate initial points using mesh-based approach.
    Start with icosphere, then select n points and build mesh.
    """
    verts, faces = _build_icosphere_mesh(n_target=n)
    
    if len(verts) == n:
        return verts
    elif len(verts) > n:
        # Select n points from the mesh
        rng = np.random.RandomState(42)
        # Farthest point sampling
        selected = [0]
        for _ in range(n - 1):
            dists = np.min([np.linalg.norm(verts - verts[i], axis=1) for i in selected], axis=0)
            for s in selected:
                dists[s] = -1
            next_idx = np.argmax(dists)
            selected.append(next_idx)
        selected = sorted(selected)
        return verts[selected]
    else:
        # Pad with Fibonacci points
        fib = _fibonacci_spiral(n)
        return fib
>>>>>>> REPLACE


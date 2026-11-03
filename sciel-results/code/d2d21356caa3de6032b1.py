<<<<<<< SEARCH
    # Add perturbed Fibonacci
    rng = np.random.RandomState(42)
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.01
        inits.append(_project_to_sphere(perturbed))
=======
    # Add perturbed Fibonacci with varying perturbation strengths
    rng = np.random.RandomState(42)
    for k in range(5):
        strength = [0.005, 0.01, 0.02, 0.05, 0.1][k]
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * strength
        inits.append(_project_to_sphere(perturbed))
    
    # Add icosahedron-based structured start
    def _icosahedron_based(n):
        """Generate points based on icosahedron symmetry."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        # Icosahedron vertices
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Generate points by rotating icosahedron vertices
        points = []
        rng2 = np.random.RandomState(123)
        for _ in range(n):
            v = verts[rng2.randint(len(verts))]
            # Small random perturbation
            v = v + rng2.randn(3) * 0.01
            v = v / np.linalg.norm(v)
            points.append(v)
        return np.array(points)
    
    inits.append(_icosahedron_based(N))
>>>>>>> REPLACE

Now let me also update the `_energy_and_grad` function to use the faster dot-product formula:


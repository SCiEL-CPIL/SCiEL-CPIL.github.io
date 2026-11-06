import numpy as np
from autoresearch_suite.chem_clusters import relax

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    def get_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        v = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        return v * 0.5

    def make_fibonacci(n, radius):
        pts = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i + 0.5) * 2 / n
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append([radius*x, radius*y, radius*z])
        return np.array(pts)

    # Initialize multiple starting configurations
    inits = []
    
    # Config 1: Icosahedral core + Fibonacci shell (Original)
    pts1 = [np.zeros(3)]
    ico1 = get_icosahedron()
    pts1.extend(ico1)
    if n > 13:
        remaining = n - len(pts1)
        fib_shell = make_fibonacci(remaining, 1.8)
        pts1.extend(fib_shell.tolist())
    inits.append(np.array(pts1))
    
    # Config 2: Pure Fibonacci (Larger radius)
    inits.append(make_fibonacci(n, 2.0))
    
    # Config 3: Random (Seeded for reproducibility)
    np.random.seed(42)
    inits.append(np.random.uniform(-2.0, 2.0, (n, 3)))
    
    # Config 4: Icosahedral core + Dodecahedral shell (if n allows)
    if n >= 33:
        pts4 = [np.zeros(3)]
        pts4.extend(ico1.tolist())
        phi = (1 + np.sqrt(5)) / 2
        # Dodecahedron vertices (Face centers of Ico)
        dodeca = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [-phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, -1/phi]
        ])
        # Scale Dodecahedron to be outside Ico shell
        # Ico radius is ~0.95. Dodeca vertices magnitude is 2.0.
        # We want them at ~1.5-2.0.
        dodeca_scaled = dodeca * 0.75 
        pts4.extend(dodeca_scaled.tolist())
        
        # Add remaining atoms using Fibonacci on a larger shell
        if n > 33:
            remaining = n - 33
            fib_extra = make_fibonacci(remaining, 2.5)
            pts4.extend(fib_extra.tolist())
        inits.append(np.array(pts4))

    best_pos = None
    best_energy = float('inf')
    
    for init in inits:
        try:
            energy, positions = relax(init, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions
        except Exception:
            continue
            
    return best_pos

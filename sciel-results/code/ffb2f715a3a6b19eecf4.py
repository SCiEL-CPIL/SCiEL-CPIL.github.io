<<<<<<< SEARCH
def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def initial_config_staggered():
    """Staggered grid arrangement."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    # 4x3 grid with stagger
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            x -= 1.5 * dx
            y -= dy
            centers.append([x, y, 0.0])
    return centers
=======
def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration for 12 hexagons."""
    centers = []
    # Hexagonal lattice: 1 center + 6 ring + 5 more
    dx = 2.0 * np.sqrt(3)  # distance between adjacent hex centers in hex lattice
    dy = 3.0               # vertical spacing
    
    # Central hexagon
    centers.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons
    for i in range(6):
        theta = np.radians(60 * i)
        x = dx * np.cos(theta)
        y = dx * np.sin(theta)
        centers.append([x, y, 0.0])
    
    # Second ring: 5 more hexagons (not all 12 to keep it compact)
    for i in range(6):
        theta = np.radians(60 * i)
        x = 2 * dx * np.cos(theta)
        y = 2 * dx * np.sin(theta)
        if len(centers) < N:
            centers.append([x, y, 0.0])
    
    return centers[:N]

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = []
    for i in range(N):
        x = rng.uniform(-4, 4)
        y = rng.uniform(-4, 4)
        a = rng.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def initial_config_staggered():
    """Staggered grid arrangement."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            x -= 1.5 * dx
            y -= dy
            centers.append([x, y, 0.0])
    return centers
>>>>>>> REPLACE


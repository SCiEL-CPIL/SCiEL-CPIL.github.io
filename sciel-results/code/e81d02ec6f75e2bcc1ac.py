<<<<<<< SEARCH
def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(hash((size, potential)) % (2 ** 31))

    candidates = []

    # --- Strategy A: divide-and-conquer assembly from polytetrahedral units ---
    for attempt in range(2):
        fragments = _random_fragments(size, rng)
        cfg, e = _assemble(fragments, rng, n_iter_optim=80, n_restarts=4)
        # final full relaxation
        e2, pos2 = _relax_full(cfg, potential, parameter)
        candidates.append((e2, pos2))

    # --- Strategy B: assemble from a few large icosahedra + fillers ---
    # Build as many 13-atom icosahedra as possible, then a filler fragment.
    n_ico = size // 13
    rem = size - 13 * n_ico
    fragments = []
    for _ in range(n_ico):
        fragments.append(_icosahedron(np.zeros(3), radius=1.0))
    if rem > 0:
        fragments.append(_random_tight_cluster(rem, rng))
    cfg, e = _assemble(fragments, rng, n_iter_optim=100, n_restarts=5)
    e2, pos2 = _relax_full(cfg, potential, parameter)
    candidates.append((e2, pos2))

    # --- Strategy C: keep the original random compact seed (safety net) ---
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.85
    for _ in range(3):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.9 * spacing for q in points):
                points.append(p)
        e3, pos3 = _relax_full(np.array(points), potential, parameter)
        candidates.append((e3, pos3))

    # pick the best
    candidates.sort(key=lambda x: x[0])
    best_e, best_pos = candidates[0]

    # final safety: ensure minimum separation of 0.5
    best_pos = _enforce_min_distance(best_pos, 0.5, rng)
    return best_pos
=======
def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(hash((size, potential)) % (2 ** 31))

    candidates = []

    # --- Strategy A: divide-and-conquer with mixed ico/deca fragments ---
    for attempt in range(3):
        mix = 0.3 + 0.4 * attempt  # vary the mix ratio
        fragments = _random_fragments(size, rng, mix_ratio=mix)
        cfg, e = _assemble(fragments, rng, n_iter_optim=100, n_restarts=5)
        if cfg is not None:
            e2, pos2 = _relax_full(cfg, potential, parameter)
            candidates.append((e2, pos2))

    # --- Strategy B: all-icosahedra + filler ---
    n_ico = size // 13
    rem = size - 13 * n_ico
    fragments = []
    for _ in range(n_ico):
        fragments.append(_icosahedron(np.zeros(3), radius=1.0))
    if rem > 0:
        fragments.append(_random_tight_cluster(rem, rng))
    cfg, e = _assemble(fragments, rng, n_iter_optim=120, n_restarts=5)
    if cfg is not None:
        e2, pos2 = _relax_full(cfg, potential, parameter)
        candidates.append((e2, pos2))

    # --- Strategy C: all-decahedra + filler ---
    fragments = []
    n_deca = size // 13
    rem = size - 13 * n_deca
    for _ in range(n_deca):
        fragments.append(_decahedron(np.zeros(3), radius=1.0))
    if rem > 0:
        fragments.append(_random_tight_cluster(rem, rng))
    cfg, e = _assemble(fragments, rng, n_iter_optim=120, n_restarts=5)
    if cfg is not None:
        e2, pos2 = _relax_full(cfg, potential, parameter)
        candidates.append((e2, pos2))

    # --- Strategy D: random compact seed (safety net) ---
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.85
    for _ in range(2):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.9 * spacing for q in points):
                points.append(p)
        e3, pos3 = _relax_full(np.array(points), potential, parameter)
        candidates.append((e3, pos3))

    # pick the best
    if not candidates:
        # Fallback: simple random
        pts = rng.normal(0, 1.5, (size, 3))
        e, pos = _relax_full(pts, potential, parameter)
        return _enforce_min_distance(pos, 0.5, rng)

    candidates.sort(key=lambda x: x[0])
    best_e, best_pos = candidates[0]

    # final safety: ensure minimum separation of 0.5
    best_pos = _enforce_min_distance(best_pos, 0.5, rng)
    return best_pos
>>>>>>> REPLACE


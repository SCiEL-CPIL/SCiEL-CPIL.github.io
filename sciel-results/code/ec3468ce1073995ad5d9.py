    # 2. Perturbed Icosahedral seed
    ico_seed = _build_icosahedral_seed(size, rng)
    perturbed = ico_seed + rng.normal(0, 0.1, (size, 3))
    candidates.append(perturbed)

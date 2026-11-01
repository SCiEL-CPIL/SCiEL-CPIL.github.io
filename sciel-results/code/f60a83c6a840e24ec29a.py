def _de_search(base, budget):
    pop_size = 10
    # Initialize population: base + perturbations
    pop = [base.copy()]
    for _ in range(pop_size - 1):
        pop.append(base + np.random.randn(*base.shape) * 0.3)
    
    # Relax all, evaluate
    energies = [_relax(p)[1] for p in pop]
    best_idx = np.argmin(energies)
    best_pos = pop[best_idx].copy()
    best_energy = energies[best_idx]
    
    # DE loop
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    while time.time() < budget:
        # For each individual, create offspring
        for i in range(pop_size):
            if time.time() > budget: break
            # Mutation: pick 3 distinct random individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            v = pop[a] + F * (pop[b] - pop[c])
            # Crossover with pop[i]
            trial = np.where(np.random.rand(*pop[i].shape) < CR, v, pop[i])
            # Relax
            trial_relaxed, e_trial = _relax(trial)
            # Selection
            if e_trial < energies[i]:
                pop[i] = trial_relaxed
                energies[i] = e_trial
                if e_trial < best_energy:
                    best_energy = e_trial
                    best_pos = trial_relaxed.copy()
    return best_pos, best_energy

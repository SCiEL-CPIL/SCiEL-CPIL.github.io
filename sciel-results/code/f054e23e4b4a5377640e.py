"""Differential Evolution + Basin Hopping for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 285.0  # Safety margin: evaluator kills at 330s

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.all(dists >= min_dist)

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:min(n-1, 12)])
    
    if n > 13:
        # Add more atoms on a sphere
        remaining = n - len(pts)
        for i in range(remaining):
            y = 1.0 - 2.0 * (i + 0.5) / remaining
            r = np.sqrt(1.0 - y*y)
            theta = 2.399963229728653 * i
            p = np.array([r*np.cos(theta), r*np.sin(theta), y]) * 2.5
            pts.append(p)
    
    return np.array(pts[:n])

def _generate_random(n, seed=0):
    """Generate a random cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        pts = rng.uniform(-3, 3, (n, 3))
        if _is_valid(pts):
            return pts
    return _generate_fcc(n)

def _energy_func(positions, kind):
    """Compute energy using the provided function."""
    try:
        e, g = sutton_chen_energy_gradient(positions, kind)
        return e
    except Exception:
        return np.inf

def _polish(positions, kind):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    def fun(x):
        try:
            e, g = sutton_chen_energy_gradient(x.reshape(-1, 3), kind)
            if not np.isfinite(e):
                return e, np.zeros_like(x)
            return e, g
        except Exception:
            return 1e10, np.zeros_like(x)
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success and np.isfinite(res.fun):
            pos = res.x.reshape(-1, 3)
            if _is_valid(pos):
                return res.fun, pos
    except Exception:
        pass
    
    # Fallback to relax
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return energy, pos
    except Exception:
        pass
    
    return _energy_func(positions, kind), positions

def _bh_step(pos, kind, step_size, rng, deadline):
    """One basin hopping step: perturb, relax, accept/reject."""
    noise = rng.normal(0, step_size, pos.shape)
    perturbed = pos + noise
    
    try:
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        if not np.isfinite(new_energy) or not _is_valid(new_pos):
            return pos, _energy_func(pos, kind), False
    except Exception:
        return pos, _energy_func(pos, kind), False
    
    return new_pos, new_energy, True

def _differential_evolution(positions_list, kind, n_gen, deadline):
    """Run differential evolution on a population of structures."""
    n_atoms = positions_list[0].shape[0]
    pop_size = len(positions_list)
    
    # Initial population
    population = [p.copy() for p in positions_list]
    energies = [_energy_func(p, kind) for p in population]
    
    # Sort by energy
    order = np.argsort(energies)
    population = [population[i] for i in order]
    energies = [energies[i] for i in order]
    
    best_energy = energies[0]
    best_pos = population[0].copy()
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    rng = np.random.default_rng(42)
    
    for gen in range(n_gen):
        if _time_left() < 30.0:
            break
        
        for i in range(pop_size):
            if _time_left() < 30.0:
                    break
            
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            
            # Crossover
            trial = np.zeros_like(mutant)
            j_rand = rng.randint(0, 3 * n_atoms)
            for j in range(3 * n_atoms):
                if j == j_rand or rng.random() < CR:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]
            
            trial = trial.reshape(-1, 3)
            
            # Evaluate trial
            if not _is_valid(trial):
                continue
            
            trial_energy, trial_pos = _polish(trial, kind)
            
            if trial_energy < energies[i]:
                population[i] = trial_pos
                energies[i] = trial_energy
                if trial_energy < best_energy:
                    best_energy = trial_energy
                    best_pos = trial_pos.copy()
        
        # Sort population
        order = np.argsort(energies)
        population = [population[i] for i in order]
        energies = [energies[i] for i in order]
    
    return best_energy, best_pos

def _bh_chain(pos, kind, n_steps, rng, deadline):
    """Run a basin hopping chain. Returns (best_energy, best_pos)."""
    best_energy = np.inf
    best_pos = pos.copy()
    
    try:
        energy, pos = relax(pos, 'sutton_chen', kind)
        if not np.isfinite(energy) or not _is_valid(pos):
            return np.inf, best_pos
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    except Exception:
        return np.inf, best_pos
    
    step_size = 0.4
    consecutive_rejects = 0
    
    for step in range(n_steps):
        if _time_left() < 30.0:
            break
        
        if consecutive_rejects > 15:
            step_size = min(step_size * 1.5, 1.5)
            consecutive_rejects = 0
        
        pos, energy, accepted = _bh_step(pos, kind, step_size, rng, deadline)
        
        if accepted and energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            step_size = max(step_size * 0.95, 0.2)
        elif not accepted:
            consecutive_rejects += 1
    
    return best_energy, best_pos

def _worker_de(args):
    """Worker for differential evolution."""
    positions_list, kind, n_gen, seed, deadline = args
    try:
        energy, pos = _differential_evolution(positions_list, kind, n_gen, deadline)
        return (energy, pos)
    except Exception:
        return (np.inf, positions_list[0])

def _worker_bh(args):
    """Worker for basin hopping."""
    pos, kind, n_steps, seed, deadline = args
    rng = np.random.default_rng(seed)
    try:
        energy, best_pos = _bh_chain(pos, kind, n_steps, rng, deadline)
        return (energy, best_pos)
    except Exception:
        return (np.inf, pos)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial structures
    initial_structures = []
    
    # FCC
    try:
        fcc = _generate_fcc(n)
        if _is_valid(fcc):
            initial_structures.append(fcc)
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _generate_bcc(n)
        if _is_valid(bcc):
            initial_structures.append(bcc)
    except Exception:
        pass
    
    # Icosahedron
    try:
        ico = _generate_icosahedron(n)
        if _is_valid(ico):
            initial_structures.append(ico)
    except Exception:
        pass
    
    # Random structures
    for seed in [1, 2, 3, 4, 5]:
        try:
            rnd = _generate_random(n, seed)
            if _is_valid(rnd):
                initial_structures.append(rnd)
        except Exception:
            pass
    
    if not initial_structures:
        initial_structures = [_generate_fcc(n)]
    
    # Phase 1: Differential Evolution with diverse population
    # Use all initial structures as the DE population
    pop_for_de = initial_structures[:min(8, len(initial_structures))]
    
    # Determine generations based on time budget
    time_budget_de = min(_time_left() - 60.0, 120.0)
    if time_budget_de < 20.0:
        time_budget_de = 20.0
    
    # Estimate: ~10-30s per generation for n~80
    n_gen_de = max(2, int(time_budget_de / 20.0))
    
    # Run DE in parallel (split population among workers)
    n_workers_de = min(4, len(pop_for_de))
    
    if n_workers_de >= 2 and len(pop_for_de) >= 2:
        try:
            # Split population among workers
            tasks_de = []
            chunk_size = max(2, len(pop_for_de) // n_workers_de)
            for w in range(n_workers_de):
                start = w * chunk_size
                end = min(start + chunk_size, len(pop_for_de))
                if start < end:
                    sub_pop = pop_for_de[start:end]
                    seed = 1000 + w * 13 + n
                    tasks_de.append((sub_pop, kind, n_gen_de, seed, _START_TIME + _DEADLINE))
            
            with mp.Pool(processes=n_workers_de) as pool:
                results_de = pool.map(_worker_de, tasks_de)
            
            for energy, pos in results_de:
                if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                    best_energy = energy
                    best_pos = pos.copy()
        except Exception:
            pass
    
    # Phase 2: Basin Hopping from best DE result and diverse starts
    # Use remaining time for BH
    time_budget_bh = min(_time_left() - 20.0, 140.0)
    if time_budget_bh < 10.0:
        time_budget_bh = 10.0
    
    # Start BH from: best DE result + top diverse starts
    bh_starts = []
    if best_pos is not None:
        bh_starts.append(best_pos.copy())
    for s in initial_structures[:4]:
        if s is not best_pos:
            bh_starts.append(s)
    
    n_workers_bh = min(4, len(bh_starts))
    if n_workers_bh > 0:
        # Estimate steps per chain
        n_steps_bh = max(10, int(time_budget_bh / (n_workers_bh * 0.5)))
        
        tasks_bh = []
        for i, start in enumerate(bh_starts[:n_workers_bh]):
            seed = 5000 + i * 17 + n
            tasks_bh.append((start, kind, n_steps_bh, seed, _START_TIME + _DEADLINE))
        
        try:
            with mp.Pool(processes=n_workers_bh) as pool:
                results_bh = pool.map(_worker_bh, tasks_bh)
            
            for energy, pos in results_bh:
                if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                    best_energy = energy
                    best_pos = pos.copy()
        except Exception:
            # Fallback: sequential BH
            for i, start in enumerate(bh_starts[:2]):
                if _time_left() < 15.0:
                    break
                rng = np.random.default_rng(5000 + i * 17 + n)
                energy, pos = _bh_chain(start, kind, n_steps_bh, rng, _START_TIME + _DEADLINE)
                if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                    best_energy = energy
                    best_pos = pos.copy()
    
    # Final polish on the best result
    if best_pos is not None and _time_left() > 5.0:
        try:
            final_energy, final_pos = _polish(best_pos, kind)
            if final_energy < best_energy and np.isfinite(final_energy) and _is_valid(final_pos):
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_pos is None:
        for s in initial_structures[:3]:
            if _time_left() < 10.0:
                break
            try:
                energy, pos = relax(s, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(pos):
                    best_energy = energy
                    best_pos = pos
                    break
            except Exception:
                continue
    
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END

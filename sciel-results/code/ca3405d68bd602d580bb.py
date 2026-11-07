"""
Strategy: Deterministic Greedy Energy-Lowering Growth (GELG).

Unlike the previous stochastic multi-start approach which relies on random
initialization and basin hopping to escape local minima, this strategy
constructs the cluster atom-by-atom. At each step, we place a new atom at the
position that minimizes the total energy of the cluster, given the current
positions of the existing atoms. This exploits the physical intuition that
stable clusters are built by adding atoms to low-energy sites (interstices,
surfaces).

The algorithm:
1. Start with a seed of 2-3 atoms in a reasonable configuration.
2. For each subsequent atom (up to n):
   a. Generate a candidate set of positions (e.g., on a grid or around existing atoms).
   b. Evaluate the energy of the cluster with each candidate added.
   c. Place the atom at the candidate with the lowest energy.
3. Relax the final configuration using L-BFGS-B.
4. If time permits, perform a few targeted basin-hopping steps to polish the result.

This is a structural change from "random search + local optimization" to
"constructive heuristics + local optimization".
"""

import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _compute_energy(positions, kind):
    """Compute Sutton-Chen energy for a given configuration."""
    try:
        energy, _ = sutton_chen_energy_gradient(positions, kind)
        return energy
    except Exception:
        return float('inf')

def _greedy_growth(n, kind, seed=0):
    """
    Grow a cluster of size n by sequentially adding atoms at the position
    that minimizes the total energy.
    """
    rng = np.random.default_rng(seed)
    
    # Initial seed: 2 atoms
    if n < 2:
        return np.zeros((n, 3))
    
    # Start with two atoms at a reasonable distance
    # Typical nearest-neighbor distance in SC is ~1.0-1.2
    d0 = 1.1
    positions = np.array([
        [-d0/2, 0, 0],
        [d0/2, 0, 0]
    ])
    
    # If n is small, just relax and return
    if n <= 2:
        return positions
    
    # For the first few atoms, use a fixed small set of candidate positions
    # to establish a stable core
    if n <= 4:
        # Try to form a triangle or tetrahedron
        candidates = []
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            candidates.append([d0 * np.cos(angle), d0 * np.sin(angle), 0])
        candidates = np.array(candidates)
        
        best_pos = None
        best_e = float('inf')
        for c in candidates:
            test_pos = np.vstack([positions, c])
            e = _compute_energy(test_pos, kind)
            if e < best_e:
                best_e = e
                best_pos = c
        positions = np.vstack([positions, best_pos])
        
        if n == 4:
            # Add 4th atom
            candidates = []
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for z in [-0.5, 0, 0.5]:
                    candidates.append([d0 * np.cos(angle), d0 * np.sin(angle), z])
            candidates = np.array(candidates)
            best_pos = None
            best_e = float('inf')
            for c in candidates:
                test_pos = np.vstack([positions, c])
                e = _compute_energy(test_pos, kind)
                if e < best_e:
                    best_e = e
                    best_pos = c
            positions = np.vstack([positions, best_pos])
            
        return positions

    # For larger n, use a more systematic candidate generation
    # We maintain a list of "candidate sites" which are positions near existing atoms
    # or on a grid surrounding the cluster.
    
    # To make this efficient, we don't evaluate ALL possible positions.
    # Instead, we generate candidates based on the current cluster geometry.
    
    for i in range(3, n):
        if _time_left() < 5:
            break
            
        # Generate candidates
        candidates = []
        
        # 1. Candidates near existing atoms (interstices)
        # For each existing atom, check positions in a shell around it
        for p in positions:
            # Simple grid around each atom
            for dx in np.linspace(-0.5, 0.5, 5):
                for dy in np.linspace(-0.5, 0.5, 5):
                    for dz in np.linspace(-0.5, 0.5, 5):
                        c = p + np.array([dx, dy, dz])
                        # Check if too close to any existing atom
                        if np.all(np.linalg.norm(positions - c, axis=1) > 0.4):
                            candidates.append(c)
                            
        # 2. Candidates on a global grid surrounding the cluster
        # Estimate cluster radius
        center = np.mean(positions, axis=0)
        radius = np.max(np.linalg.norm(positions - center, axis=1))
        
        grid_step = 0.4
        max_coord = int(radius / grid_step) + 2
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                for z in range(-max_coord, max_coord + 1):
                    c = center + np.array([x, y, z]) * grid_step
                    if np.all(np.linalg.norm(positions - c, axis=1) > 0.4):
                        candidates.append(c)
        
        if not candidates:
            # Fallback: random
            c = center + rng.normal(0, 0.5, 3)
            candidates = [c]
            
        candidates = np.array(candidates)
        
        # Evaluate candidates
        # To save time, we can subsample if too many
        if len(candidates) > 200:
            idx = np.random.choice(len(candidates), 200, replace=False)
            candidates = candidates[idx]
            
        best_pos = None
        best_e = float('inf')
        
        # We can vectorize energy calculation for speed, but sutton_chen_energy_gradient
        # takes a single configuration. So we loop.
        for c in candidates:
            test_pos = np.vstack([positions, c])
            e = _compute_energy(test_pos, kind)
            if e < best_e:
                best_e = e
                best_pos = c
                
        if best_pos is not None:
            positions = np.vstack([positions, best_pos])
        else:
            # Should not happen
            break
            
    return positions

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Run Greedy Growth with a few different seeds to account for
    #    the stochastic nature of the candidate selection (if we use random subsampling)
    #    and initial conditions.
    
    for seed in range(3):
        if _time_left() < 10:
            break
            
        # Generate initial configuration via greedy growth
        try:
            pos = _greedy_growth(n, kind, seed=seed)
            
            # Relax the configuration
            energy, relaxed_pos = relax(pos, potential, kind)
            
            if np.isfinite(energy):
                # Check validity
                min_d = np.inf
                for i in range(len(relaxed_pos)):
                    for j in range(i+1, len(relaxed_pos)):
                        d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                        if d < min_d:
                            min_d = d
                            
                if min_d >= 0.35:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed_pos
        except Exception:
            continue
            
    # 2. If we have a valid result, try to improve it with a short basin hopping
    #    to escape any local minima found by the greedy construction.
    
    if best_positions is not None and _time_left() > 20:
        current_best = best_positions.copy()
        current_energy = best_energy
        step_size = 0.1
        rng = np.random.default_rng(123)
        
        # Short, aggressive basin hopping
        max_hops = 50
        for hop in range(max_hops):
            if _time_left() < 5:
                break
                
            noise = rng.normal(0, step_size, (n, 3))
            perturbed = current_best + noise
            
            try:
                energy, positions = relax(perturbed, potential, kind)
                if np.isfinite(energy):
                    min_d = np.inf
                    for i in range(len(positions)):
                        for j in range(i+1, len(positions)):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_d:
                                min_d = d
                                
                    if min_d >= 0.35:
                        if energy < current_energy:
                            current_energy = energy
                            current_best = positions
                            step_size *= 0.9
                        else:
                            step_size *= 1.1
                            step_size = min(step_size, 0.3)
            except Exception:
                continue
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_best
            
    # 3. Fallback if greedy growth failed or produced invalid results
    if best_positions is None:
        # Simple random start + relax
        rng = np.random.default_rng(0)
        radius = n ** (1/3)
        pos = rng.uniform(-radius, radius, (n, 3))
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if np.isfinite(energy):
                min_d = np.inf
                for i in range(len(relaxed_pos)):
                    for j in range(i+1, len(relaxed_pos)):
                        d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    best_positions = relaxed_pos
        except Exception:
            # Last resort: grid
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i, 0, 0] * 0.5 # Spread them out
            
    return best_positions

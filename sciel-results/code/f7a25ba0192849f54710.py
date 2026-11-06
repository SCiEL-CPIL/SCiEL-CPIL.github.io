"""HP lattice protein folding using Memetic Algorithm (Genetic Algorithm + Local Search)."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import defaultdict


def _neighbours(point):
    """Yield the 4 neighbours of a point in 2D lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain, where=None, h_positions=None):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    if where is None or h_positions is None:
        where = {tuple(p): i for i, p in enumerate(chain)}
        h_positions = {tuple(p) for i, p in enumerate(chain) if sequence[i] == 'H'}
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            pt = tuple(p)
            for q in _neighbours(pt):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _build_index(chain):
    return {tuple(p): i for i, p in enumerate(chain)}


def _is_valid_chain(chain):
    """Check if chain is self-avoiding and connected."""
    seen = set()
    for p in chain:
        if tuple(p) in seen:
            return False
        seen.add(tuple(p))
    for i in range(1, len(chain)):
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            return False
    return True


def _initial_spiral_walk(n, rng):
    """Generate a compact spiral-like initial walk."""
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dx, dy = 1, 0
    steps = 1
    turns = 0
    while len(chain) < n:
        for _ in range(steps):
            if len(chain) >= n:
                break
            nx, ny = chain[-1][0] + dx, chain[-1][1] + dy
            if (nx, ny) in occupied:
                break
            chain.append((nx, ny))
            occupied.add((nx, ny))
        if len(chain) >= n:
            break
        # Turn left
        dx, dy = -dy, dx
        turns += 1
        if turns % 2 == 0:
            steps += 1
    # If spiral failed to reach n, extend randomly
    while len(chain) < n:
        moved = False
        for _ in range(10):
            nx, ny = chain[-1][0] + rng.choice([-1, 1, 0, 0]), chain[-1][1] + rng.choice([0, 0, -1, 1])
            if rng.random() < 0.5:
                nx, ny = chain[-1][0] + rng.choice([-1, 1]), chain[-1][1]
            else:
                nx, ny = chain[-1][0], chain[-1][1] + rng.choice([-1, 1])
            if (nx, ny) not in occupied:
                chain.append((nx, ny))
                occupied.add((nx, ny))
                moved = True
                break
        if not moved:
            break
    return [list(p) for p in chain[:n]]


def _initial_random_compact(n, rng):
    """Generate a random walk that tries to stay compact."""
    chain = [(0, 0)]
    occupied = {(0, 0)}
    for _ in range(1, n):
        options = []
        for nx, ny in _neighbours(chain[-1]):
            if (nx, ny) not in occupied:
                # Bias towards staying near origin
                dist = abs(nx) + abs(ny)
                options.append((nx, ny, dist))
        if not options:
            break
        # Choose option with smallest distance with some probability
        if rng.random() < 0.7:
            min_dist = min(o[2] for o in options)
            best = [o for o in options if o[2] == min_dist]
            choice = rng.choice(best)
        else:
            choice = rng.choice(options)
        nx, ny = choice[0], choice[1]
        chain.append((nx, ny))
        occupied.add((nx, ny))
    return [list(p) for p in chain[:n]]


def _local_search(sequence, chain, rng, time_limit=0.1):
    """Fast stochastic local search to improve a single chain."""
    n = len(chain)
    start_time = time.time()
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = [list(p) for p in chain]
    
    T = 1.0
    
    while time.time() - start_time < time_limit:
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.8:
            # Reversal
            if n > 2:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
        else:
            # Step move: pick a residue (not ends) and try to move it to a neighbor
            # This is a "kink" move or "step" move
            i = rng.randint(1, n - 2)
            prev_pt = tuple(chain[i-1])
            curr_pt = tuple(chain[i])
            next_pt = tuple(chain[i+1])
            
            # Try moving chain[i] to a neighbor of prev_pt or next_pt?
            # Actually, standard move is to pick i, remove it, and insert it elsewhere? 
            # Or just try to move the step from (i-1, i) to a different direction?
            # Let's do a "step flip" or "neighbor swap"
            # Simpler: Try to move chain[i] to a free neighbor of chain[i-1] that is not chain[i-2]
            # This changes the path at i.
            
            options = []
            for nx, ny in _neighbours(prev_pt):
                if (nx, ny) != curr_pt and (nx, ny) != tuple(chain[i-2]) if i > 1 else True:
                    # Check if it collides with rest of chain
                    # We need to check if (nx, ny) is free in the chain excluding i
                    # And if chain[i+1] is adjacent to (nx, ny) ? No, chain[i+1] is fixed.
                    # If we move chain[i], chain[i-1] -> new_pos -> chain[i+1] must be valid.
                    # So new_pos must be adjacent to chain[i-1] AND chain[i+1].
                    # This is a "reconnection" or "flip".
                    if abs(nx - next_pt[0]) + abs(ny - next_pt[1]) == 1:
                        # Check collision
                        collides = False
                        for k, p in enumerate(chain):
                            if k != i and tuple(p) == (nx, ny):
                                collides = True
                                break
                        if not collides:
                            new_chain = chain[:]
                            new_chain[i] = [nx, ny]
                            break
            if new_chain is None:
                continue

        if new_chain is None:
            continue
            
        # Validate
        if not _is_valid_chain(new_chain):
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in chain]
            T *= 0.99
        else:
            T = max(0.1, T * 0.95)
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Memetic Algorithm for HP folding."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    total_budget = 15.0  # seconds
    start_time = time.time()
    
    # Initialize population
    pop_size = 10
    population = []
    energies = []
    
    for i in range(pop_size):
        if i < pop_size // 2:
            chain = _initial_spiral_walk(n, random.Random(i * 123))
        else:
            chain = _initial_random_compact(n, random.Random(i * 456))
        
        # Quick local search
        chain, energy = _local_search(sequence, chain, random.Random(999 + i), time_limit=0.1)
        
        if len(chain) == n:
            population.append(chain)
            energies.append(energy)
    
    if not population:
        return [[i, 0] for i in range(n)]
    
    best_chain = min(population, key=lambda c: _compute_energy(sequence, c))
    best_energy = min(energies)
    
    # Evolutionary Loop
    rng = random.Random(42)
    while time.time() - start_time < total_budget:
        # Select parents (Tournament Selection)
        parent1_idx = rng.randint(0, len(population) - 1)
        parent2_idx = rng.randint(0, len(population) - 1)
        
        # Crossover: Simple splice
        # Find a cut point where the chains can be reconnected
        # This is complex. Let's use "Recombination by Splice" if possible, else Mutation.
        # Given time constraints, Mutation is safer and effective in Memetic Algos.
        # Let's do a "Hybrid" approach: 
        # 1. Take best individual, apply strong local search.
        # 2. Take random individual, apply mutation, local search.
        
        # Strategy: 
        # - Keep the best individual.
        # - Generate new candidates by mutating random individuals from the population.
        # - Replace the worst individual in the population if the new one is better.
        
        # Pick a random individual to mutate
        idx = rng.randint(0, len(population) - 1)
        parent = population[idx]
        
        # Mutation: Random Reversal
        child = parent[:]
        if n > 2:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            child[i:j+1] = child[i:j+1][::-1]
        
        # Validate child
        if not _is_valid_chain(child):
            # If invalid, try a different mutation: Step Move
            child = parent[:]
            if n > 2:
                k = rng.randint(1, n - 2)
                prev_pt = tuple(child[k-1])
                next_pt = tuple(child[k+1])
                # Try to find a new position for child[k]
                for nx, ny in _neighbours(prev_pt):
                    if (nx, ny) != tuple(child[k]) and abs(nx - next_pt[0]) + abs(ny - next_pt[1]) == 1:
                        collides = False
                        for m, p in enumerate(child):
                            if m != k and tuple(p) == (nx, ny):
                                collides = True
                                break
                        if not collides:
                            child[k] = [nx, ny]
                            break
        
        if not _is_valid_chain(child):
            continue
            
        # Local Search on child
        child, energy = _local_search(sequence, child, rng, time_limit=0.2)
        
        if len(child) != n:
            continue
            
        # Replace worst in population
        worst_idx = max(range(len(population)), key=lambda i: energies[i])
        
        if energy < energies[worst_idx]:
            population[worst_idx] = child
            energies[worst_idx] = energy
            
            if energy < best_energy:
                best_energy = energy
                best_chain = [list(p) for p in child]
                
        # Also try to improve the current best directly with more time
        if time.time() - start_time < total_budget - 0.5:
            improved_best, improved_energy = _local_search(sequence, best_chain, rng, time_limit=0.5)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_best
                # Update population with this improved best
                population[0] = best_chain
                energies[0] = best_energy
                population.sort(key=lambda c: _compute_energy(sequence, c))
                energies.sort()
                
    return best_chain
# EVOLVE-BLOCK-END

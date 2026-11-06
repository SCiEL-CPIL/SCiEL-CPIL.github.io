"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

_DEADLINE = time.time() + 310  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=0.0):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + bias * rng.random() + (1 - bias) * rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, i, rng, dimension):
    """Try a pivot move at position i: rotate the tail segment by 90 degrees around an axis."""
    if i <= 1 or i >= len(chain) - 1:
        return None
    
    n = len(chain)
    head = chain[:i+1]
    tail = chain[i+1:]
    
    axis = rng.randint(0, dimension - 1)
    sign = rng.choice([-1, 1])
    
    pivot = chain[i]
    new_tail = []
    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        new_tail.append(new_pos)
    
    new_chain = head + new_tail
    
    occupied = set(new_chain)
    if len(occupied) != n:
        return None
    
    for k in range(n - 1):
        diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
        if diff != 1:
            return None
    
    return new_chain


def _local_search(sequence, chain, dimension, rng, max_iters=200):
    """Fast local search: accept only improving moves."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move(current, i, rng, dimension)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        # else: reject (steepest descent style)
    
    return best, best_energy


def _anneal(sequence, chain, dimension, rng, max_iters=400):
    """Simulated annealing refinement of a chain."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    T_start = 3.0
    T_end = 0.005
    if max_iters > 1:
        alpha = (T_end / T_start) ** (1.0 / max_iters)
    else:
        alpha = 1.0
    T = T_start
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move(current, i, rng, dimension)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        T *= alpha
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    n = len(sequence)
    # Determine number of restarts based on chain length and time budget
    if n <= 48:
        n_restarts = 200
    elif n <= 80:
        n_restarts = 120
    else:
        n_restarts = 70
    
    # Phase 1: Quick local search on multiple restarts
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 60:
            break
        bias = 0.3 + 0.7 * (r / max(1, n_restarts - 1))
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=150)
        
        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    # Phase 2: Deep annealing on the best few solutions found
    if best is not None and time.time() < _DEADLINE - 30:
        # Take the current best and do a deeper anneal
        refined, refined_energy = _anneal(sequence, best, dimension, rng, max_iters=500)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
        
        # Also try annealing from a couple of fresh starts
        for extra in range(3):
            if time.time() > _DEADLINE - 10:
                break
            chain = _grow(sequence, dimension, rng, bias=0.5)
            if chain is None:
                continue
            refined, refined_energy = _anneal(sequence, chain, dimension, rng, max_iters=400)
            if refined_energy < best_energy:
                best, best_energy = refined, refined_energy
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

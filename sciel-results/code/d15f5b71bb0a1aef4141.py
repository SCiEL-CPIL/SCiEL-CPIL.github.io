"""
HP lattice protein folding - Parallel Tempering Monte Carlo strategy.
This approach uses a fixed self-avoiding walk scaffold and explores
conformational space via Metropolis acceptance in multiple temperature
replicas, allowing the system to escape local minima more effectively
than incremental growth or simple local search.
"""
import time
import random
import sys
import math

# Global time tracking
_START_TIME = time.time()
_TIME_BUDGET = 290  # seconds, leaving 10s safety margin before 300s limit


def _remaining_time():
    return max(0.0, _TIME_BUDGET - (time.time() - _START_TIME))


def _neighbours(point):
    """Generate all 6 neighbours in 3D cubic lattice."""
    n = len(point)
    for axis in range(n):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy for a chain. Energy = -number of H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _random_grow(sequence, dimension, rng, bias=0.3):
    """Random chain growth with slight bias toward hydrophobic contacts."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count potential H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free space (escape from dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * bias + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Weighted selection
        total = sum(c[0] for c in candidates)
        r = rng.random() * total
        cum = 0
        chosen = candidates[-1][1]
        for score, site in candidates:
            cum += score
            if r <= cum:
                chosen = site
                break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _parallel_tempering(sequence, chain, occupied, current_energy, time_limit=20.0, num_replicas=5):
    """
    Parallel Tempering Monte Carlo on the chain.
    Uses multiple temperature replicas to escape local minima.
    """
    n = len(chain)
    start_time = time.time()
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    # Initialize replicas with different temperatures
    # T ranges from low (greedy) to high (exploratory)
    T_min = 0.5
    T_max = 10.0
    temps = [T_min + (T_max - T_min) * i / (num_replicas - 1) for i in range(num_replicas)]
    
    # Each replica has its own chain and energy
    replicas = []
    for t in temps:
        # Start from the current best chain
        rep_chain = list(chain)
        rep_occupied = {p: i for i, p in enumerate(rep_chain)}
        rep_energy = current_energy
        replicas.append({
            'chain': rep_chain,
            'occupied': rep_occupied,
            'energy': rep_energy,
            'T': t
        })
    
    best_energy = current_energy
    best_chain = list(chain)
    
    while (time.time() - start_time) < time_limit:
        # Perform one Monte Carlo step for each replica
        for rep_idx in range(num_replicas):
            rep = replicas[rep_idx]
            rep_chain = rep['chain']
            rep_occupied = rep['occupied']
            rep_energy = rep['energy']
            T = rep['T']
            
            # Pick a random internal residue to move
            i = rng.randint(1, n - 2)  # Avoid endpoints
            pos = rep_chain[i]
            
            # Try moving to a nearby empty site
            targets = list(_neighbours(pos))
            rng.shuffle(targets)
            
            moved = False
            for target in targets:
                if target in rep_occupied:
                    continue
                
                # Check adjacency constraints
                if i > 0:
                    d1 = sum(abs(a - b) for a, b in zip(target, rep_chain[i-1]))
                    if d1 != 1:
                        continue
                if i < n - 1:
                    d2 = sum(abs(a - b) for a, b in zip(target, rep_chain[i+1]))
                    if d2 != 1:
                        continue
                
                # Compute energy delta
                old_gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        j = rep_occupied.get(q)
                        if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                            old_gain += 1
                        elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                            old_gain += 1
                
                new_gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(target):
                        j = rep_occupied.get(q)
                        if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                            new_gain += 1
                        elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                            new_gain += 1
                
                delta = old_gain - new_gain  # positive means energy increases (worse)
                new_energy = rep_energy + delta
                
                # Metropolis acceptance
                if new_energy <= rep_energy or rng.random() < math.exp(-delta / T):
                    # Accept
                    del rep_occupied[pos]
                    rep_chain[i] = target
                    rep_occupied[target] = i
                    rep_energy = new_energy
                    moved = True
                    
                    # Update global best
                    if rep_energy < best_energy:
                        best_energy = rep_energy
                        best_chain = list(rep_chain)
                    break
            
            # Update replica state
            rep['chain'] = rep_chain
            rep['occupied'] = rep_occupied
            rep['energy'] = rep_energy
        
        # Attempt swap between adjacent replicas (parallel tempering)
        for rep_idx in range(num_replicas - 1):
            if rng.random() < 0.5:  # 50% chance to attempt swap
                rep1 = replicas[rep_idx]
                rep2 = replicas[rep_idx + 1]
                
                # Compute acceptance probability for swap
                # P(accept) = min(1, exp(-(E1 - E2) * (1/T1 - 1/T2)))
                E1, T1 = rep1['energy'], rep1['T']
                E2, T2 = rep2['energy'], rep2['T']
                
                if T1 == T2:
                    continue
                
                exponent = -(E1 - E2) * (1.0/T1 - 1.0/T2)
                if exponent > 0:
                    exponent = min(exponent, 700)  # Prevent overflow
                
                if rng.random() < math.exp(exponent):
                    # Swap chains and energies
                    rep1['chain'], rep2['chain'] = rep2['chain'], rep1['chain']
                    rep1['occupied'], rep2['occupied'] = rep2['occupied'], rep1['occupied']
                    rep1['energy'], rep2['energy'] = rep2['energy'], rep1['energy']
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold an HP sequence into a self-avoiding walk on the cubic lattice."""
    global _START_TIME
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0  # 0 means unfolded (no contacts)
    
    # Phase 1: Multiple random growths to get diverse starting points
    num_growths = 20
    rng_main = random.Random(12345 + n)
    
    initial_chains = []
    for _ in range(num_growths):
        chain = _random_grow(sequence, dimension, rng_main, bias=0.3)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            occupied = {p: i for i, p in enumerate(chain)}
            initial_chains.append((chain, occupied, energy))
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Phase 2: Parallel Tempering on the best few initial chains
    # Sort by energy and take top 3
    initial_chains.sort(key=lambda x: x[2])
    top_chains = initial_chains[:min(3, len(initial_chains))]
    
    for chain, occupied, energy in top_chains:
        if _remaining_time() < 10.0:
            break
        
        # Parallel tempering
        pt_time = min(30.0, _remaining_time() * 0.6)
        chain, energy = _parallel_tempering(sequence, chain, occupied, energy, time_limit=pt_time, num_replicas=5)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Phase 3: Final polish with a short parallel tempering run on the best chain
    if _remaining_time() > 5.0 and best_chain is not None:
        occupied = {p: i for i, p in enumerate(best_chain)}
        pt_time = min(10.0, _remaining_time() * 0.8)
        chain, energy = _parallel_tempering(sequence, best_chain, occupied, best_energy, time_limit=pt_time, num_replicas=3)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Convert to required format
    if best_chain is None:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]

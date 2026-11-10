"""HP lattice protein folding - Parallel Tempering Monte Carlo on fixed scaffold."""
import time
import random
import math

# Global time tracking
_START_TIME = time.time()
_TIME_BUDGET = 290  # seconds, leaving 40s safety margin before 330s limit

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
    """Compute H-H contact energy for a chain."""
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

def _random_scaffold(n, dimension, rng):
    """Generate a random self-avoiding walk of length n."""
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site not in occupied:
                # Simple bias: prefer sites that don't trap the chain
                # Count free neighbors
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                # Avoid dead ends (free <= 1) unless necessary
                if free > 1 or len(candidates) == 0:
                    candidates.append((free, site))
        
        if not candidates:
            # Backtrack or fail? For simplicity, if stuck, return None
            # But random walks usually have space. 
            # If stuck, we might need a more sophisticated generator.
            # For now, try a different random seed or just fail.
            # Actually, let's just pick any valid site if available.
            valid_sites = [site for site in _neighbours(last) if site not in occupied]
            if not valid_sites:
                return None
            chosen = valid_sites[rng.randint(0, len(valid_sites)-1)]
        else:
            # Weighted choice by free space
            total_weight = sum(w for w, _ in candidates)
            r = rng.random() * total_weight
            cum = 0
            chosen = candidates[-1][1]
            for w, site in candidates:
                cum += w
                if r <= cum:
                    chosen = site
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain

def _move_residue(sequence, chain, occupied, i, target, current_energy):
    """
    Attempt to move residue i to target.
    Returns (success, new_energy).
    """
    n = len(chain)
    pos = chain[i]
    
    # Check if target is occupied
    if target in occupied:
        return False, current_energy
    
    # Check adjacency constraints with neighbors
    if i > 0:
        d1 = sum(abs(a - b) for a, b in zip(target, chain[i-1]))
        if d1 != 1:
            return False, current_energy
    if i < n - 1:
        d2 = sum(abs(a - b) for a, b in zip(target, chain[i+1]))
        if d2 != 1:
            return False, current_energy
    
    # Compute energy delta
    # Remove contacts involving i at old position
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(pos):
            j = occupied.get(q)
            if j is not None and j != i:
                if j > i + 1 and sequence[j] == 'H':
                    old_contacts += 1
                elif j < i - 1 and sequence[j] == 'H':
                    old_contacts += 1
    
    # Add contacts involving i at new position
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(target):
            j = occupied.get(q)
            if j is not None and j != i:
                if j > i + 1 and sequence[j] == 'H':
                    new_contacts += 1
                elif j < i - 1 and sequence[j] == 'H':
                    new_contacts += 1
    
    delta = old_contacts - new_contacts
    new_energy = current_energy + delta
    
    return True, new_energy

def _parallel_tempering(sequence, n, dimension, rng, time_limit):
    """
    Parallel Tempering Monte Carlo.
    """
    # Generate initial scaffolds for each replica
    num_replicas = 5
    temperatures = [0.1, 0.5, 1.0, 2.0, 4.0]
    
    replicas = []
    for t in temperatures:
        scaffold = _random_scaffold(n, dimension, rng)
        if scaffold is None:
            # Fallback: straight line
            scaffold = [(i, 0, 0) for i in range(n)]
        
        chain = [list(p) for p in scaffold]
        occupied = {tuple(p): i for i, p in enumerate(chain)}
        energy = _compute_energy(sequence, chain)
        
        replicas.append({
            'chain': chain,
            'occupied': occupied,
            'energy': energy,
            'T': t
        })
    
    start_time = time.time()
    step_count = 0
    
    while (time.time() - start_time) < time_limit:
        # Perform a move in each replica
        for rep in replicas:
            if (time.time() - start_time) >= time_limit:
                break
            
            chain = rep['chain']
            occupied = rep['occupied']
            current_energy = rep['energy']
            T = rep['T']
            
            # Pick a random residue to move (not 0 or n-1 to keep ends fixed? 
            # Actually, moving ends is fine as long as self-avoidance holds.
            # But moving residue 0 breaks the "start" anchor if we care about it.
            # In SAW, the chain is just a set of connected points.
            # Let's allow moving any residue except 0 to keep the start fixed for simplicity.
            i = rng.randint(1, n - 1)
            
            pos = chain[i]
            
            # Generate candidate moves
            # 1. Rotate/Move to a neighbor of current position
            # 2. Move to a neighbor of a neighbor (2-step jump) - might be too aggressive
            # Let's stick to 1-step moves first, but try all 6 neighbors.
            
            moved = False
            candidates = list(_neighbours(pos))
            rng.shuffle(candidates)
            
            for target in candidates:
                target_t = tuple(target)
                if target_t in occupied:
                    continue
                
                valid, new_energy = _move_residue(sequence, chain, occupied, i, target_t, current_energy)
                if not valid:
                    continue
                
                delta = new_energy - current_energy
                
                # Metropolis acceptance
                if delta <= 0:
                    accept = True
                else:
                    # Avoid overflow
                    if T == 0:
                        accept = False
                    else:
                        accept_prob = math.exp(-delta / T)
                        accept = rng.random() < accept_prob
                
                if accept:
                    # Perform move
                    del occupied[pos]
                    chain[i] = list(target_t)
                    occupied[target_t] = i
                    rep['energy'] = new_energy
                    moved = True
                    break
            
            step_count += 1
        
        # Attempt replica exchange
        # Sort replicas by energy? Or just try random pairs?
        # Standard PT: try to exchange adjacent replicas in temperature ladder.
        # For simplicity, try to exchange every pair with probability.
        # Or just swap the two replicas with highest and lowest energy?
        # Let's try swapping adjacent replicas in the list (which are sorted by T).
        
        # Sort replicas by T to ensure adjacent exchange logic works
        # The list is already in order of T because we created it that way.
        
        for k in range(num_replicas - 1):
            rep1 = replicas[k]
            rep2 = replicas[k + 1]
            
            T1 = rep1['T']
            T2 = rep2['T']
            E1 = rep1['energy']
            E2 = rep2['energy']
            
            # Delta for exchange
            # P(accept) = min(1, exp(-(E1/T2 - E1/T1 - E2/T1 + E2/T2)))
            # = min(1, exp(-((E1 - E2) * (1/T2 - 1/T1))))
            
            delta_T = 1.0 / T2 - 1.0 / T1
            delta_E = E1 - E2
            exponent = -delta_E * delta_T
            
            if exponent >= 0:
                swap = True
            else:
                # Clamp exponent to avoid underflow
                if exponent < -100:
                    swap = False
                else:
                    swap = rng.random() < math.exp(exponent)
            
            if swap:
                # Swap chains and energies
                rep1['chain'], rep2['chain'] = rep2['chain'], rep1['chain']
                rep1['occupied'], rep2['occupied'] = rep2['occupied'], rep1['occupied']
                rep1['energy'], rep2['energy'] = rep2['energy'], rep1['energy']
    
    # Find the best replica (lowest energy)
    best_rep = min(replicas, key=lambda r: r['energy'])
    return best_rep['chain'], best_rep['energy']

def fold(sequence, dimension):
    """Fold an HP sequence into a self-avoiding walk on the cubic lattice."""
    global _START_TIME
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + n)
    
    # Allocate time for PT
    # PT is slow, so we use most of the budget.
    time_for_pt = _remaining_time() * 0.9
    
    if time_for_pt < 1.0:
        # Not enough time, just return a random scaffold
        chain = _random_scaffold(n, dimension, rng)
        if chain is None:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
        return [list(p) for p in chain]
    
    best_chain, best_energy = _parallel_tempering(sequence, n, dimension, rng, time_for_pt)
    
    # Final check: ensure self-avoidance and valid steps
    # (PT should maintain this, but just in case)
    occupied = set()
    valid = True
    for i, p in enumerate(best_chain):
        if tuple(p) in occupied:
            valid = False
            break
        occupied.add(tuple(p))
        if i > 0:
            prev = best_chain[i-1]
            d = sum(abs(a-b) for a,b in zip(p, prev))
            if d != 1:
                valid = False
                break
    
    if not valid:
        # Fallback
        chain = _random_scaffold(n, dimension, rng)
        if chain is None:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
        best_chain = chain
    
    return [list(p) for p in best_chain]

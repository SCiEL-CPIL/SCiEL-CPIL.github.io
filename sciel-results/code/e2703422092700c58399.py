"""HP lattice protein folding with Genetic Algorithm and Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _get_neighbours_fast(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y + z + 1, z - 1) # Wait, typo in thought process, fix below
    ]

def _get_neighbours_fast(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _get_neighbours_fast(*p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _get_neighbours_fast(*chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _get_neighbours_fast(*site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _get_neighbours_fast(*site) if q not in occupied)
            score = gain * 5.0 + free * 0.5 + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

class ChainState:
    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.chain = list(chain)
        self.n = len(chain)
        self.pos_map = {p: i for i, p in enumerate(self.chain)}
        self.energy = _energy(sequence, self.chain)
        self.h_indices = [i for i in range(self.n) if sequence[i] == "H"]

    def copy(self):
        return ChainState(self.sequence, self.chain)

    def delta_flip(self, i, new_pos):
        """Calculate energy delta for moving residue i to new_pos."""
        old_pos = self.chain[i]
        if old_pos == new_pos:
            return 0
        
        is_h = self.sequence[i] == "H"
        if not is_h:
            return 0
        
        # Count H-H contacts involving residue i at old_pos
        old_contacts = 0
        for q in _get_neighbours_fast(*old_pos):
            j = self.pos_map.get(q)
            if j is not None and abs(j - i) > 1 and self.sequence[j] == "H":
                old_contacts += 1
        
        # Count H-H contacts involving residue i at new_pos
        new_contacts = 0
        for q in _get_neighbours_fast(*new_pos):
            j = self.pos_map.get(q)
            if j is not None and abs(j - i) > 1 and self.sequence[j] == "H":
                new_contacts += 1
        
        return -(new_contacts - old_contacts)

    def apply_flip(self, i, new_pos):
        old_pos = self.chain[i]
        self.pos_map.pop(old_pos)
        self.pos_map[new_pos] = i
        self.chain[i] = new_pos
        self.energy += self.delta_flip(i, new_pos)

    def valid_flip_options(self, i):
        """Get valid positions for a flip move at index i."""
        old_pos = self.chain[i]
        prev = self.chain[i - 1]
        nxt = self.chain[i + 1]
        options = []
        for new_pos in _get_neighbours_fast(*old_pos):
            d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
            d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
            if d1 != 1 or d2 != 1:
                continue
            if new_pos in self.pos_map:
                continue
            options.append(new_pos)
        return options

def _anneal(state, rng, time_budget):
    """Simulated Annealing refinement."""
    n = state.n
    if n < 4:
        return state

    start_time = time.time()
    best_state = state.copy()
    best_energy = state.energy
    current_state = state
    current_energy = state.energy
    
    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.1
    
    # Estimate steps
    est_steps_per_sec = 15000
    total_steps = int(time_budget * est_steps_per_sec)
    total_steps = max(100, min(total_steps, 20000))
    
    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0
    
    for step in range(total_steps):
        if time.time() - start_time >= time_budget - 0.1:
            break
        
        T *= cooling_factor
        if T < T_min:
            T = T_min
        
        i = rng.randint(1, n - 2)
        options = current_state.valid_flip_options(i)
        
        if not options:
            continue
            
        new_pos = rng.choice(options)
        delta = current_state.delta_flip(i, new_pos)
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current_state.apply_flip(i, new_pos)
            
            if current_state.energy < best_energy:
                best_energy = current_state.energy
                best_state = current_state.copy()
                
    return best_state

def _crossover(parent1, parent2, rng):
    """Splice crossover. Tries to merge prefix of parent1 with suffix of parent2."""
    n = parent1.n
    # Choose a cut point
    cut = rng.randint(1, n - 2)
    
    prefix1 = parent1.chain[:cut + 1]
    suffix2 = parent2.chain[cut + 1:]
    
    # We need to check if the junction is valid and if the whole chain is self-avoiding
    # Simple check: last of prefix1 and first of suffix2 must be neighbors
    p_last = prefix1[-1]
    s_first = suffix2[0]
    d = abs(p_last[0] - s_first[0]) + abs(p_last[1] - s_first[1]) + abs(p_last[2] - s_first[2])
    if d != 1:
        return None
    
    new_chain = prefix1 + suffix2
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
        
    return ChainState(parent1.sequence, new_chain)

def fold(sequence, dimension):
    """
    Genetic Algorithm with Simulated Annealing.
    """
    deadline = time.time() + 235.0  # Safety margin
    n = len(sequence)
    
    # Initial population
    pop_size = 10
    population = []
    
    for i in range(pop_size):
        rng = random.Random(1000 + i * 7919 + len(sequence))
        chain = _grow(sequence, dimension, rng)
        if chain:
            population.append(ChainState(sequence, chain))
            
    if not population:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Sort initial population
    population.sort(key=lambda s: s.energy)
    
    best_state = population[0].copy()
    best_energy = best_state.energy
    
    # Main GA Loop
    generation = 0
    while time.time() < deadline - 10.0:
        generation += 1
        
        # Adaptive population size: shrink if time is running out
        if time.time() > deadline - 60.0:
            current_pop_size = 5
        else:
            current_pop_size = pop_size
            
        # Selection and Crossover
        new_population = []
        
        # Elitism: keep the best 2
        for i in range(min(2, len(population))):
            new_population.append(population[i].copy())
            
        # Generate offspring
        while len(new_population) < current_pop_size:
            # Tournament selection
            p1 = rng.choice(population) if 'rng' in locals() else random.choice(population)
            p2 = rng.choice(population) if 'rng' in locals() else random.choice(population)
            
            # Use a persistent rng for consistency within a run? Or new one?
            # Let's use a global-ish rng for the GA loop
            if not hasattr(fold, '_ga_rng'):
                fold._ga_rng = random.Random(42)
            ga_rng = fold._ga_rng
            
            p1 = ga_rng.choice(population)
            p2 = ga_rng.choice(population)
            
            child = _crossover(p1, p2, ga_rng)
            
            if child is None:
                # Mutation: copy parent and random flip
                parent = ga_rng.choice([p1, p2])
                child = parent.copy()
                # Random mutation
                for _ in range(5):
                    i = ga_rng.randint(1, n - 2)
                    options = child.valid_flip_options(i)
                    if options:
                        new_pos = ga_rng.choice(options)
                        delta = child.delta_flip(i, new_pos)
                        if delta <= 0 or ga_rng.random() < 0.1: # Accept worse with low prob
                            child.apply_flip(i, new_pos)
            else:
                # Mutate child slightly
                for _ in range(3):
                    i = ga_rng.randint(1, n - 2)
                    options = child.valid_flip_options(i)
                    if options:
                        new_pos = ga_rng.choice(options)
                        delta = child.delta_flip(i, new_pos)
                        if delta <= 0 or ga_rng.random() < 0.05:
                            child.apply_flip(i, new_pos)
                        
            new_population.append(child)
            
        # Anneal each new individual
        # Budget for annealing: remaining time / (num_individuals * generations_estimate)
        # We don't know generations estimate, so we just allocate a fixed small budget per individual
        # and check time
        time_per_anneal = 0.5 # seconds
        
        annealed_population = []
        for child in new_population:
            if time.time() > deadline - 5.0:
                annealed_population.append(child)
                continue
                
            # Anneal
            annealed = _anneal(child, ga_rng, time_per_anneal)
            annealed_population.append(annealed)
            
        # Combine and select
        population = annealed_population
        population.sort(key=lambda s: s.energy)
        
        if population[0].energy < best_energy:
            best_energy = population[0].energy
            best_state = population[0].copy()
            
    # Final refinement of the best state
    if time.time() < deadline - 2.0:
        final_budget = deadline - time.time() - 2.0
        final_state = _anneal(best_state, fold._ga_rng, final_budget)
        if final_state.energy < best_energy:
            best_state = final_state
            best_energy = final_state.energy

    return [list(p) for p in best_state.chain]
# EVOLVE-BLOCK-END

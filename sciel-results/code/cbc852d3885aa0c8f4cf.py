# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_routes(n=10):
    """Precompute which edges each demand uses for CW and CCW routing."""
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _compute_loads(routing, cw_edges, ccw_edges, demands):
    """Compute edge loads given a routing vector."""
    n = 10
    D = len(demands)
    loads = np.zeros(n)
    
    for d in range(D):
        s, t, w = demands[d]
        if routing[d] >= 0.5:
            for e in ccw_edges[d]:
                loads[e] += w
        else:
            for e in cw_edges[d]:
                loads[e] += w
    
    return loads


def _compute_max_load(routing, cw_edges, ccw_edges, demands):
    """Compute just the maximum load for a routing."""
    loads = _compute_loads(routing, cw_edges, ccw_edges, demands)
    return np.max(loads)


def _local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=100000):
    """Local search with 1-flip moves."""
    D = len(demands)
    best_routing = routing.copy()
    best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d in range(D):
            trial = best_routing.copy()
            trial[d] = 1.0 - trial[d]
            trial_max = _compute_max_load(trial, cw_edges, ccw_edges, demands)
            if trial_max < best_max - 1e-9:
                best_routing = trial
                best_max = trial_max
                improved = True
    
    return best_routing, best_max


def _pair_local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=50000):
    """Local search with 2-flip moves."""
    D = len(demands)
    best_routing = routing.copy()
    best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d1 in range(D):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, D):
                trial = best_routing.copy()
                trial[d1] = 1.0 - trial[d1]
                trial[d2] = 1.0 - trial[d2]
                trial_max = _compute_max_load(trial, cw_edges, ccw_edges, demands)
                if trial_max < best_max - 1e-9:
                    best_routing = trial
                    best_max = trial_max
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    return best_routing, best_max


def _simulated_annealing(routing, cw_edges, ccw_edges, demands, deadline, T0=1.0, cooling=0.9995, max_iter=200000):
    """Simulated annealing for escaping local optima."""
    D = len(demands)
    current_routing = routing.copy()
    current_max = _compute_max_load(current_routing, cw_edges, ccw_edges, demands)
    best_routing = current_routing.copy()
    best_max = current_max
    
    T = T0
    np_random = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random demand to flip
        d = np_random.integers(0, D)
        trial = current_routing.copy()
        trial[d] = 1.0 - trial[d]
        trial_max = _compute_max_load(trial, cw_edges, ccw_edges, demands)
        
        delta = trial_max - current_max
        if delta < 0 or np_random.random() < np.exp(-delta / max(T, 1e-10)):
            current_routing = trial
            current_max = trial_max
            if current_max < best_max - 1e-9:
                best_routing = current_routing.copy()
                best_max = current_max
        
        T *= cooling
    
    return best_routing, best_max


def _genetic_algorithm(cw_edges, ccw_edges, demands, deadline, pop_size=50, max_gen=500):
    """Simple genetic algorithm."""
    D = len(demands)
    np_random = np.random.default_rng(123)
    
    # Initialize population
    pop = np.zeros((pop_size, D))
    for i in range(pop_size):
        pop[i] = np_random.random(D)
    
    def fitness(r):
        return -_compute_max_load(r, cw_edges, ccw_edges, demands)
    
    best_routing = None
    best_max = float('inf')
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fits = np.array([fitness(r) for r in pop])
        idx = np.argmax(fits)
        if fits[idx] > -best_max + 1e-9:
            best_max = -fits[idx]
            best_routing = pop[idx].copy()
        
        # Selection (tournament)
        new_pop = np.zeros_like(pop)
        new_pop[0] = pop[idx]  # Elitism
        
        for i in range(1, pop_size):
            # Tournament selection
            contenders = np_random.choice(pop_size, 3, replace=False)
            winner = contenders[np.argmax(fits[contenders])]
            new_pop[i] = pop[winner].copy()
        
        # Crossover and mutation
        for i in range(1, pop_size // 2):
            p1_idx = i
            p2_idx = pop_size - 1 - i
            if np_random.random() < 0.7:  # Crossover probability
                point = np_random.integers(1, D)
                child1 = np.concatenate([pop[p1_idx][:point], pop[p2_idx][point:]])
                child2 = np.concatenate([pop[p2_idx][:point], pop[p1_idx][point:]])
            else:
                child1 = pop[p1_idx].copy()
                child2 = pop[p2_idx].copy()
            
            # Mutation
            for j in range(D):
                if np_random.random() < 0.1:
                    child1[j] = 1.0 - child1[j]
                if np_random.random() < 0.1:
                    child2[j] = 1.0 - child2[j]
            
            new_pop[p1_idx] = child1
            new_pop[p2_idx] = child2
        
        pop = new_pop
    
    if best_routing is None:
        best_routing = pop[0].copy()
        best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    return best_routing, best_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using a combination of
    local search, simulated annealing, and genetic algorithm.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    best_routing = None
    best_max = float('inf')
    
    # Phase 1: Multiple random restarts with local search
    np_random = np.random.default_rng(2024)
    
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        # Random initial solution
        routing = np_random.random(D)
        
        # Local search
        lr, lm = _local_search(routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=50000)
        if lm < best_max - 1e-9:
            best_max = lm
            best_routing = lr.copy()
    
    # Phase 2: Pair local search from best
    if best_routing is not None and time.time() < deadline - 20:
        pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=20000)
        if pm < best_max - 1e-9:
            best_max = pm
            best_routing = pr.copy()
    
    # Phase 3: Simulated annealing from best
    if best_routing is not None and time.time() < deadline - 15:
        sa_r, sa_m = _simulated_annealing(best_routing, cw_edges, ccw_edges, demands, deadline - 5, T0=0.5, cooling=0.9998, max_iter=100000)
        if sa_m < best_max - 1e-9:
            best_max = sa_m
            best_routing = sa_r.copy()
    
    # Phase 4: Genetic algorithm
    if time.time() < deadline - 10:
        ga_r, ga_m = _genetic_algorithm(cw_edges, ccw_edges, demands, deadline - 2, pop_size=40, max_gen=200)
        if ga_m < best_max - 1e-9:
            best_max = ga_m
            best_routing = ga_r.copy()
    
    # Phase 5: Final local search polish
    if best_routing is not None and time.time() < deadline - 2:
        final_r, final_m = _local_search(best_routing, cw_edges, ccw_edges, demands, deadline, max_iter=100000)
        if final_m < best_max - 1e-9:
            best_max = final_m
            best_routing = final_r.copy()
    
    if best_routing is None:
        best_routing = np.zeros(D)
    
    return best_routing


# EVOLVE-BLOCK-END

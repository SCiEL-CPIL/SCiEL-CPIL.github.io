"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import math


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            a, b = chosen[i], chosen[j]
            d_pa = _dist_sq(p, a)
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _make_maximal(chosen, all_points, rng):
    """Greedily add points to chosen to make it maximal, maintaining independence."""
    chosen = list(chosen)
    remaining = [p for p in all_points if p not in chosen]
    rng.shuffle(remaining)
    
    for p in remaining:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
            
    return chosen


def _local_search(start_set, all_points, rng, max_iters=5000):
    """Simulated Annealing / Local Search to minimize size of maximal set."""
    current_set = _make_maximal(start_set, all_points, rng)
    current_size = len(current_set)
    best_set = list(current_set)
    best_size = current_size
    
    T = 1.0
    cooling = 0.999
    
    for _ in range(max_iters):
        # Generate a neighbor
        if not current_set:
            break
            
        # Strategy: Remove a random point from current_set, then re-maximize
        # This effectively explores different "bases" for the maximal set
        remove_idx = rng.randint(0, len(current_set) - 1)
        new_set_base = current_set[:remove_idx] + current_set[remove_idx+1:]
        
        # Re-maximize from this base
        new_set = _make_maximal(new_set_base, all_points, rng)
        new_size = len(new_set)
        
        # Acceptance criterion
        if new_size < current_size:
            current_set = new_set
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_set)
        else:
            # Metropolis criterion
            delta = new_size - current_size
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_set = new_set
                    current_size = new_size
        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_set


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with different seeds
    for seed in range(20):
        rng = random.Random(seed)
        # Initialize with a small random subset to allow local search to grow it
        # Or just start with empty and let _make_maximal do the work?
        # Starting with empty gives a specific maximal set. 
        # To get diversity, we can start with a few random points.
        init_size = rng.randint(0, 3)
        init_points = rng.sample(all_points, init_size)
        
        result = _local_search(init_points, all_points, rng, max_iters=2000)
        
        if len(result) < best_size:
            best_size = len(result)
            best_set = result
            
    return best_set
# EVOLVE-BLOCK-END

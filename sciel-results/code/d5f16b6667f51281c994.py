import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    # Precompute all 4-cycles as arrays of edge indices
    # Each cycle is a tuple of 4 edge indices
    all_cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))
        
    # Convert to numpy array for vectorization
    # Shape: (num_cycles, 4)
    cycles_arr = np.array(all_cycles, dtype=np.int32)
    num_cycles = cycles_arr.shape[0]
    
    # Precompute a mask or just use the array directly.
    # We will evaluate "violations" as the count of cycles where all 4 colors are distinct.
    # A cycle is valid if it is NOT the case that c0!=c1, c0!=c2, c0!=c3, c1!=c2, c1!=c3, c2!=c3.
    # It is easier to count violations directly in a vectorized way?
    # Let's define a function that takes a color vector and returns the number of violations.
    
    def count_violations(colors):
        """
        Returns the number of 4-cycles with 4 distinct colors.
        colors: np.array of shape (120,)
        """
        # Get colors for all cycles at once
        # cycle_colors shape: (num_cycles, 4)
        cycle_colors = colors[cycles_arr]
        
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        
        # A cycle is a violation if all are distinct.
        # distinct_01 = c0 != c1
        # distinct_02 = c0 != c2
        # distinct_03 = c0 != c3
        # distinct_12 = c1 != c2
        # distinct_13 = c1 != c3
        # distinct_23 = c2 != c3
        # violation = distinct_01 & distinct_02 & distinct_03 & distinct_12 & distinct_13 & distinct_23
        
        # This involves 6 boolean arrays of size num_cycles.
        # num_cycles is 1680. This is very fast in numpy.
        
        v = (c0 != c1) & (c0 != c2) & (c0 != c3) & \
            (c1 != c2) & (c1 != c3) & \
            (c2 != c3)
            
        return np.sum(v)

    def get_score(colors):
        """
        Objective: Maximize distinct colors.
        Constraint: Violations must be 0.
        We use a penalty method or lexicographic objective.
        Primary: Minimize violations.
        Secondary: Maximize distinct colors.
        """
        v = count_violations(colors)
        if v > 0:
            # Return a negative score to penalize invalid states
            # We want to minimize v, then maximize distinct colors.
            # Let's return -v * 1000 + distinct_count? 
            # Actually, for SA, we need a single energy.
            # Energy = v * LARGE_PENALTY - distinct_count
            # We want to MINIMIZE Energy.
            # So if v=0, Energy = -distinct_count (lower is better).
            # If v>0, Energy = v * 1000 - distinct_count (very high, bad).
            distinct = len(np.unique(colors))
            return v * 1000.0 - distinct
        
        distinct = len(np.unique(colors))
        return -distinct

    # Initial state: Random assignment
    # To start close to valid, we can try a few random inits or a structured one.
    # All 0s is valid (0 violations, 1 color).
    # Let's start with random to explore, but keep a valid fallback.
    
    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_energy = 0.0 # -1 for all 0s
    
    # Try a few random initializations to see if we can start with more colors
    # But random usually has high violations.
    # We will rely on SA to fix violations.
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Simulated Annealing
    # State: colors vector
    # Move: Pick a random subset of edges to recolor? Or single edge?
    # Single edge moves are slow to escape. 
    # "Swaps" or "Blocks" are better.
    # Let's try: Pick 2 edges, swap their colors. Or Pick 1 edge, assign random color.
    # Given the constraint structure, recoloring a single edge affects ~180 cycles.
    # Vectorized check is ~1680 * 6 ops. ~10k ops.
    # 50,000 iterations * 10k ops = 500M ops. Might be too slow in Python/Numpy overhead.
    # Numpy overhead is per call. 50k calls to count_violations might take ~5-10s.
    # This is acceptable.
    
    # Let's optimize count_violations further?
    # The boolean ops are vectorized. The overhead is the Python function call and array indexing.
    
    # Strategy: 
    # 1. Start with all 0s.
    # 2. SA loop.
    #    - Generate move: Randomly select K edges (K=1 or 2) and assign random colors from 0..119.
    #    - Calculate new energy.
    #    - Accept if better or with prob exp(-delta/T).
    
    # To maximize colors, we should bias moves towards using NEW colors or diverse colors.
    # However, SA on the energy landscape (penalty - distinct) will naturally push for distinct colors once violations are 0.
    
    # Let's run SA for a fixed number of steps or time.
    
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    current_energy = get_score(current_colors) # 0 * 1000 - 1 = -1
    
    best_colors = current_colors.copy()
    best_energy = current_energy
    
    T = 10.0
    cooling_rate = 0.9995
    
    steps = 0
    max_steps = 200000 # Adjust based on speed
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Generate move
        # Option 1: Single edge recolor
        # Option 2: Swap colors of two edges
        # Option 3: Recolor a random block of edges
        
        # Single edge recolor is simplest and allows fine-tuning.
        # To help escape local optima, we can occasionally do a "restart" or "perturbation".
        
        if rng.random() < 0.1:
            # Perturbation: Recolor 5 random edges
            edges_to_change = np_rng.choice(NUM_EDGES, size=5, replace=False)
            new_colors = np_rng.integers(0, MAX_COLORS, size=5)
            old_vals = current_colors[edges_to_change]
            current_colors[edges_to_change] = new_colors
        else:
            # Standard move: Recolor 1 edge
            e = np_rng.integers(0, NUM_EDGES)
            new_c = np_rng.integers(0, MAX_COLORS)
            if current_colors[e] == new_c:
                continue
            old_val = current_colors[e]
            current_colors[e] = new_c
            edges_to_change = [e]
            old_vals = [old_val]
            
        new_energy = get_score(current_colors)
        
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = current_colors.copy()
        else:
            # Revert
            if isinstance(edges_to_change, np.ndarray):
                current_colors[edges_to_change] = old_vals
            else:
                current_colors[edges_to_change[0]] = old_vals[0]
                
        T *= cooling_rate
        if T < 0.01:
            # Reheat or restart?
            # Let's just stop or reset T if we want to explore more.
            # Given time budget, let's just let it cool.
            pass
            
    # Final check and cleanup
    # Ensure best_colors is valid
    # If best_colors has violations, we need a valid fallback.
    # The all-zero coloring is always valid.
    # So if best_energy corresponds to violations > 0, we should return zeros?
    # Wait, get_score returns v*1000 - distinct.
    # If v=0, score is -distinct (negative).
    # If v>0, score is positive (large).
    # So lower score is better.
    # If best_energy > 0, it means we have violations.
    # In that case, we should return the all-zero coloring (score -1) or any valid coloring found.
    # Since we start with all zeros (score -1), best_energy will be <= -1 if we ever find a valid coloring with >= 1 color.
    # If we never find a valid coloring with >0 colors (impossible, 0 is valid), we return zeros.
    
    # Check if best solution is valid
    if count_violations(best_colors) > 0:
         # Fallback to all zeros
         return np.zeros(NUM_EDGES, dtype=np.int32).tolist()
         
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
    print(colors)

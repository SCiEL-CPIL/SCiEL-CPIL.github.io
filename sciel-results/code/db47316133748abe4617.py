import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (0-based indices)
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # Precompute for each position: the indices of APs containing it, and the column index within that AP
    # This allows O(1) vectorized delta calculation
    pos_ap_indices = [[] for _ in range(N)]
    pos_ap_cols = [[] for _ in range(N)]
    
    for i in range(total_aps):
        for j in range(4):
            p = aps[i, j]
            pos_ap_indices[p].append(i)
            pos_ap_cols[p].append(j)
            
    # Convert to numpy arrays for fast access
    pos_ap_indices_arr = [np.array(idx, dtype=np.int32) for idx in pos_ap_indices]
    pos_ap_cols_arr = [np.array(cols, dtype=np.int8) for cols in pos_ap_cols]

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy."""
        c = colors[aps]
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_compute_delta(colors, p):
        """
        Compute the change in mono count if we flip color at position p.
        Uses precomputed indices and vectorized numpy operations.
        """
        ap_idxs = pos_ap_indices_arr[p]
        cols = pos_ap_cols_arr[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0
        
        # Get the colors of the other 3 elements in each AP? 
        # Actually, it's easier to get the full 4-tuple, modify the specific column, and compare.
        # But we can be smarter.
        # An AP is mono if all 4 are equal.
        # Let old_color = colors[p].
        # For each AP containing p:
        #   Before: mono if all 4 == old_color (since p is old_color).
        #   After:  mono if all 4 == 1-old_color (since p becomes 1-old_color).
        
        # Let's extract the colors of the 3 *other* elements for each AP.
        # This is tricky with standard indexing without knowing the other positions.
        # Let's stick to the robust method:
        # 1. Get sub_aps (n_aps, 4)
        # 2. Get sub_colors (n_aps, 4)
        # 3. Check mono_before
        # 4. Flip the specific column in a copy
        # 5. Check mono_after
        
        sub_aps = aps[ap_idxs] # (n_aps, 4)
        sub_colors = colors[sub_aps] # (n_aps, 4)
        
        mono_before = (sub_colors.max(axis=1) == sub_colors.min(axis=1))
        
        # Create a copy to modify
        # We need to flip the column specified by 'cols' for each row
        # rows = 0..n_aps-1
        # cols = specific column for that row
        rows = np.arange(n_aps)
        sub_colors_after = sub_colors.copy()
        sub_colors_after[rows, cols] ^= 1
        
        mono_after = (sub_colors_after.max(axis=1) == sub_colors_after.min(axis=1))
        
        return int(mono_after.sum()) - int(mono_before.sum())

    def local_search(colors, max_time=1.0):
        """Hill climbing: flip bits that reduce mono count."""
        current_count = count_mono(colors)
        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False
            # Random order to break symmetry/ties
            order = np.random.permutation(N)
            for p in order:
                if (time.time() - start_ls) >= max_time:
                    break
                delta = fast_compute_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
        return current_count

    def simulated_annealing(colors, max_time=2.0):
        """
        SA to escape local minima.
        Returns best colors found during SA.
        """
        current_count = count_mono(colors)
        best_count = current_count
        best_colors = colors.copy()
        
        start_sa = time.time()
        T = 5.0  # Initial temperature
        T_min = 0.1
        
        # Estimate steps per second to adjust cooling? 
        # Let's just do fixed steps or time-based cooling.
        # Time-based cooling is safer.
        
        while (time.time() - start_sa) < max_time:
            p = random.randint(0, N - 1)
            delta = fast_compute_delta(colors, p)
            
            if delta <= 0:
                colors[p] ^= 1
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            else:
                # Accept worse with probability exp(-delta / T)
                elapsed = time.time() - start_sa
                fraction = elapsed / max_time
                T = T * (1.0 - fraction) + T_min * fraction
                if random.random() < np.exp(-delta / T):
                    colors[p] ^= 1
                    current_count += delta
            
        return best_colors, best_count

    # --- Initialization ---
    pop_size = 30
    population = []
    
    # Diverse seeds
    # 1. Random
    for _ in range(10):
        population.append(np.random.randint(0, 2, size=N).astype(np.int8))
        
    # 2. Structured patterns
    # Alternating
    population.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    # Blocks of 4
    population.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tm[i] = bin(i).count('1') % 2
    population.append(tm)
    
    # Fill rest with random perturbations of existing seeds
    while len(population) < pop_size:
        base = random.choice(population[:5])
        child = base.copy()
        n_flips = random.randint(10, 30)
        flip_pos = np.random.choice(N, size=n_flips, replace=False)
        child[flip_pos] ^= 1
        population.append(child)
    
    # Initial Refinement: Short LS on all to get them to local minima
    scores = []
    for i in range(pop_size):
        # Use a very short time for initial refinement to spread time
        local_search(population[i], max_time=0.2)
        scores.append(count_mono(population[i]))
    
    best_global_count = min(scores)
    best_global_colors = population[scores.index(best_global_count)].copy()
    
    # --- Main Loop ---
    generation = 0
    no_improve_count = 0
    island_size = 5
    num_islands = pop_size // island_size
    
    while time.time() < deadline - 10:
        generation += 1
        
        # Sort population
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_global_count:
            best_global_count = scores[0]
            best_global_colors = population[0].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Create new generation
        new_population = []
        new_scores = []
        
        # Elitism: Keep top 5
        for i in range(5):
            new_population.append(population[i].copy())
            new_scores.append(scores[i])
        
        # Fill rest
        while len(new_population) < pop_size:
            # Selection: Tournament
            candidates = random.sample(range(pop_size), 3)
            best_idx = min(candidates, key=lambda i: scores[i])
            parent = population[best_idx].copy()
            
            # Mutation
            n_mutations = random.randint(2, 10)
            mut_pos = np.random.choice(N, size=n_mutations, replace=False)
            parent[mut_pos] ^= 1
            
            # Refinement: SA + LS
            # If stagnating, increase SA time to escape
            sa_time = 0.5 if no_improve_count < 5 else 1.5
            ls_time = 0.5
            
            # Check time budget
            if time.time() > deadline - 15:
                break
                
            sa_colors, sa_count = simulated_annealing(parent, max_time=sa_time)
            ls_count = local_search(sa_colors, max_time=ls_time)
            
            new_population.append(sa_colors)
            new_scores.append(ls_count)
            
        population = new_population
        scores = new_scores
        
        # If stagnating heavily, inject random individuals to reset diversity
        if no_improve_count > 10:
            for i in range(pop_size - 5):
                if random.random() < 0.2: # Replace some non-elites with random
                    population[i] = np.random.randint(0, 2, size=N).astype(np.int8)
                    local_search(population[i], max_time=0.2)
                    scores[i] = count_mono(population[i])
            no_improve_count = 0

    # Final Polish on the best solution found
    if time.time() < deadline - 5:
        local_search(best_global_colors, max_time=5.0)
        # One last SA check
        if time.time() < deadline - 2:
            sa_colors, sa_count = simulated_annealing(best_global_colors, max_time=2.0)
            ls_count = local_search(sa_colors, max_time=1.0)
            if ls_count < count_mono(best_global_colors):
                best_global_colors = sa_colors

    return [int(c) for c in best_global_colors]

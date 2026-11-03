import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute neighbor lists for each node for faster local search
    neighbors = [np.where(conflict[i] == 1)[0] for i in range(n)]
    
    rng = np.random.default_rng(42)
    
    def greedy_construct():
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        incompat = np.zeros(n, dtype=np.int32)
        for v in sel_indices:
            for nb in neighbors[v]:
                incompat[nb] += 1
        
        improved = True
        max_moves = 500
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                mask_in_S = selected == 1
                
                cond_0 = (incompat == 0)
                cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                
                sub_conflict_check = sub_conflict.copy()
                np.fill_diagonal(sub_conflict_check, 1)
                
                if (sub_conflict_check == 0).any():
                    zero_rows, zero_cols = np.where(sub_conflict_check == 0)
                    if len(zero_rows) > 0:
                        u_local = zero_rows[0]
                        w_local = zero_cols[0]
                        u = cand_indices[u_local]
                        w = cand_indices[w_local]
                        
                        for nb in neighbors[v]:
                            incompat[nb] -= 1
                        
                        for nb in neighbors[u]:
                            incompat[nb] += 1
                        
                        for nb in neighbors[w]:
                            incompat[nb] += 1
                        
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        moves_made += 1
                        break
            
            if not improved:
                if k_sel > 0:
                    v_idx_in_S = rng.integers(k_sel)
                    v = sel_indices[v_idx_in_S]
                    
                    mask_in_S = selected == 1
                    cond_0 = (incompat == 0)
                    cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                    Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                    
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) > 0:
                        u = cand_indices[rng.integers(len(cand_indices))]
                        
                        for nb in neighbors[v]:
                            incompat[nb] -= 1
                        for nb in neighbors[u]:
                            incompat[nb] += 1
                        
                        selected[v] = 0
                        selected[u] = 1
                        improved = True
                        moves_made += 1
                        break
        
        return selected, moves_made

    def repair(selected):
        """Remove conflicting vertices until independent set is valid."""
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            sel_indices = np.where(selected == 1)[0]
            if len(sel_indices) == 0:
                break
            for i in range(len(sel_indices)):
                v = sel_indices[i]
                if selected[v] == 0:
                    continue
                for nb in neighbors[v]:
                    if selected[nb] == 1:
                        # Remove the one with higher index to be deterministic-ish
                        # or random choice
                        if rng.random() < 0.5:
                            selected[v] = 0
                        else:
                            selected[nb] = 0
                        changed = True
                        break
        return selected

    # Initialize population with greedy solutions
    pop_size = 200
    population = []
    for i in range(pop_size):
        if time.time() > deadline - 30.0:
            break
        sel = greedy_construct()
        sel, _ = local_search(sel)
        population.append(sel)
    
    if not population:
        sel = greedy_construct()
        sel, _ = local_search(sel)
        population = [sel]
    
    best_count = max(p.sum() for p in population)
    best_selected = max(population, key=lambda x: x.sum())
    
    # Genetic Algorithm
    # Chromosome is a bitset (np.array of uint8)
    # Crossover: Uniform crossover with repair
    # Mutation: Random flip with repair
    
    max_gens = 500
    tournament_size = 5
    mutation_rate = 0.05
    crossover_rate = 0.7
    
    for gen in range(max_gens):
        if time.time() > deadline - 10.0:
            break
            
        # Evaluate fitness
        fitnesses = [p.sum() for p in population]
        
        # Update best
        max_fit = max(fitnesses)
        if max_fit > best_count:
            best_count = max_fit
            best_selected = population[fitnesses.index(max_fit)].copy()
            
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep top 10%
        elite_count = max(1, pop_size // 10)
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break
                
            # Select parents via tournament
            parent1_idx = rng.integers(pop_size)
            best_in_tour = parent1_idx
            best_fit = fitnesses[parent1_idx]
            for _ in range(tournament_size - 1):
                cand_idx = rng.integers(pop_size)
                if fitnesses[cand_idx] > best_fit:
                    best_in_tour = cand_idx
                    best_fit = fitnesses[cand_idx]
            parent1 = population[best_in_tour]
            
            parent2_idx = rng.integers(pop_size)
            best_in_tour = parent2_idx
            best_fit = fitnesses[parent2_idx]
            for _ in range(tournament_size - 1):
                cand_idx = rng.integers(pop_size)
                if fitnesses[cand_idx] > best_fit:
                    best_in_tour = cand_idx
                    best_fit = fitnesses[cand_idx]
            parent2 = population[best_in_tour]
            
            # Crossover or Mutation
            if rng.random() < crossover_rate:
                # Uniform crossover
                mask = rng.random(n) < 0.5
                child = np.zeros(n, dtype=np.uint8)
                child[mask] = parent1[mask]
                child[~mask] = parent2[~mask]
            else:
                child = parent1.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                # Flip a few random bits
                num_flips = rng.integers(1, 10)
                flip_indices = rng.choice(n, size=num_flips, replace=False)
                child[flip_indices] = 1 - child[flip_indices]
            
            # Repair: Ensure independent set
            child = repair(child)
            
            # Local search refinement (limited)
            child, _ = local_search(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Check convergence
        if len(set(tuple(p) for p in population)) < 10:
            # Diversify
            for i in range(min(10, pop_size)):
                population[i] = greedy_construct()
                population[i], _ = local_search(population[i])
    
    # Final local search on best
    best_selected, _ = local_search(best_selected)
    best_count = best_selected.sum()
    
    # Extract Result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")

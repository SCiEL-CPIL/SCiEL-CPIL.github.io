import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process to return the best prize collected by any ant.
    This is the fitness function for our meta-optimization.
    """
    n = len(prizes)
    # Ensure heuristic is positive as per contract
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0
            total_prize = prizes[0]
            
            while True:
                # Find valid next nodes
                remaining_budget = max_len - path_len
                # Condition: d[current_node, j] + d[j, 0] <= remaining_budget
                valid_indices = np.where(
                    (~visited_mask) & 
                    (distance_matrix[current_node, :] + distance_matrix[:, 0] <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Calculate probabilities
                probs = H[current_node, valid_indices]
                total_prob = np.sum(probs)
                if total_prob == 0:
                    break
                
                probs = probs / total_prob
                
                # Choose next node
                next_idx_in_valid = rng.choice(len(valid_indices), p=probs)
                next_node = valid_indices[next_idx_in_valid]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        
        # Pheromone update logic:
        # tau_new = 0.9 * tau_old + (best_prize_iter / 10)
        # Since tau starts uniform and update is affine with constant addition,
        # tau remains uniform for all i,j.
        # Thus relative probabilities are determined solely by H.
        
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin (300s limit)
    
    n = len(prizes)
    eps = 1e-9
    
    # Initial heuristic: Standard prize/distance ratio
    # H[i,j] = prize[j] / (d[i,j] + small)
    # We initialize a population based on this, but with random perturbations.
    
    # Base heuristic for initialization
    d = distance_matrix + 0.01
    p = prizes[None, :] + 0.01
    base_h = p / d
    base_h = np.where(base_h > 0, base_h, 1e-9)
    
    # Population size
    pop_size = 20
    # Initialize population
    population = []
    for i in range(pop_size):
        # Add random noise to base heuristic to create diversity
        # Multiplicative noise to maintain scale
        noise = np.random.uniform(0.5, 1.5, size=(n, n))
        h = base_h * noise
        h = np.where(h > 0, h, 1e-9)
        population.append(h)
    
    # Evaluate initial population
    fitnesses = []
    for h in population:
        score = _simulate_aco(prizes, distance_matrix, max_len, h)
        fitnesses.append(score)
    
    best_h = population[0]
    best_score = fitnesses[0]
    for i in range(1, pop_size):
        if fitnesses[i] > best_score:
            best_score = fitnesses[i]
            best_h = population[i]
    
    # Genetic Algorithm Loop
    # We will run generations until time runs out.
    # Each generation:
    # 1. Select parents (Tournament selection)
    # 2. Crossover (Blend crossover / Linear interpolation)
    # 3. Mutation (Gaussian noise on log-scale or direct additive)
    # 4. Evaluate new individuals
    # 5. Replace worst in population
    
    pop = population[:]
    fit = fitnesses[:]
    
    while time.time() < deadline:
        # Check time budget for a full generation
        # Estimate cost: pop_size simulations.
        # If we've already done 20 sims in ~1s, one gen is ~1s.
        
        new_individuals = []
        new_fitnesses = []
        
        # Number of new individuals to generate (replace half the population)
        num_offspring = pop_size // 2
        
        for _ in range(num_offspring):
            if time.time() > deadline - 5: # Stop if close to deadline
                break
                
            # 1. Selection: Tournament of size 3
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            idx3 = np.random.randint(0, pop_size)
            
            candidates = [(idx1, fit[idx1]), (idx2, fit[idx2]), (idx3, fit[idx3])]
            candidates.sort(key=lambda x: x[1], reverse=True)
            parent1 = pop[candidates[0][0]]
            
            # Second parent: second best of tournament or random from top half
            # Let's just take the second best of the tournament for simplicity
            parent2 = pop[candidates[1][0]]
            
            # 2. Crossover: Linear blend (Lerp)
            alpha = np.random.uniform(0.0, 1.0)
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # 3. Mutation
            # Additive Gaussian noise, scaled by the magnitude of the matrix
            # Log-space mutation is often better for positive quantities, 
            # but let's stick to additive for simplicity and stability, 
            # clamped to positive.
            # Scale noise by mean of parent
            scale = np.mean(parent1)
            noise = np.random.normal(0, 0.1 * scale, size=(n, n))
            child = child + noise
            
            # Ensure positivity
            child = np.where(child > 0, child, 1e-9)
            
            new_individuals.append(child)
            
            # 4. Evaluation
            score = _simulate_aco(prizes, distance_matrix, max_len, child)
            new_fitnesses.append(score)
            
            if score > best_score:
                best_score = score
                best_h = child
        
        if not new_individuals:
            break
            
        # 5. Replacement
        # Combine old and new, sort by fitness, keep top pop_size
        combined_pop = pop + new_individuals
        combined_fit = fit + new_fitnesses
        
        # Sort indices by fitness descending
        indices = np.argsort(combined_fit)[::-1]
        
        pop = [combined_pop[i] for i in indices[:pop_size]]
        fit = [combined_fit[i] for i in indices[:pop_size]]
        
        # Update global best if found in this batch (already done in loop, but double check)
        if fit[0] > best_score:
            best_score = fit[0]
            best_h = pop[0]

    return best_h

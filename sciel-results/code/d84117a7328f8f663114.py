"""
Program to find a tournament on n vertices where every set of 4 vertices has a common dominator.
Uses a genetic algorithm with NumPy-accelerated cost evaluation.
Falls back to the known Paley tournament on n=67 if no better solution is found within the time limit.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit
    
    # Known best solution: Paley tournament on p=67
    p_known = 67
    squares = {x * x % p_known for x in range(1, p_known)}
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)
    
    best_n = p_known
    best_orientation = orientation_known
    
    # Try to find a solution for n = 66
    target_n = 66
    m = target_n * (target_n - 1) // 2
    
    # Helper to map (i, j) to index in orientation array
    # i < j
    # Index = i*(target_n - 1) - i*(i+1)//2 + (j - i - 1)
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-tuples as a numpy array for vectorized operations
    # Each row is (a, b, c, d) with a < b < c < d
    quad_sets = []
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                for d in range(c + 1, target_n):
                    quad_sets.append((a, b, c, d))
    
    quad_sets = np.array(quad_sets, dtype=np.int32)
    num_quads = len(quad_sets)
    
    # Precompute edge indices for each 4-set
    # For a set (a, b, c, d), the 6 edges are:
    # (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
    # We need the indices in the orientation array for each edge.
    edge_indices = np.zeros((num_quads, 6), dtype=np.int32)
    for i in range(num_quads):
        a, b, c, d = quad_sets[i]
        edge_indices[i, 0] = get_idx(a, b, target_n)
        edge_indices[i, 1] = get_idx(a, c, target_n)
        edge_indices[i, 2] = get_idx(a, d, target_n)
        edge_indices[i, 3] = get_idx(b, c, target_n)
        edge_indices[i, 4] = get_idx(b, d, target_n)
        edge_indices[i, 5] = get_idx(c, d, target_n)
    
    # Precompute which vertex is the candidate dominator for each edge in each 4-set
    # For each 4-set, we need to check if any vertex dominates the other 3.
    # We can vectorize this by checking all 4 possible dominators for each 4-set.
    
    def evaluate_cost(orient, n):
        """
        orient: numpy array of 0/1, shape (m,)
        Returns the number of bad 4-sets (sets with no common dominator).
        """
        # orient[e] = 0 means i->j, 1 means j->i for edge (i,j) where i<j.
        # For a 4-set (a, b, c, d):
        #   a dominates b,c,d if orient[idx(a,b)]=0, orient[idx(a,c)]=0, orient[idx(a,d)]=0
        #   b dominates a,c,d if orient[idx(a,b)]=1 (b->a), orient[idx(b,c)]=0, orient[idx(b,d)]=0
        #   c dominates a,b,d if orient[idx(a,c)]=1, orient[idx(b,c)]=1, orient[idx(c,d)]=0
        #   d dominates a,b,c if orient[idx(a,d)]=1, orient[idx(b,d)]=1, orient[idx(c,d)]=1
        
        # Let's extract the relevant bits for all 4-sets at once.
        # edge_indices[i, k] gives the edge index for the k-th edge in the i-th 4-set.
        # Edges: 0:(a,b), 1:(a,c), 2:(a,d), 3:(b,c), 4:(b,d), 5:(c,d)
        
        e_ab = orient[edge_indices[:, 0]]  # 0: a->b, 1: b->a
        e_ac = orient[edge_indices[:, 1]]  # 0: a->c, 1: c->a
        e_ad = orient[edge_indices[:, 2]]  # 0: a->d, 1: d->a
        e_bc = orient[edge_indices[:, 3]]  # 0: b->c, 1: c->b
        e_bd = orient[edge_indices[:, 4]]  # 0: b->d, 1: d->b
        e_cd = orient[edge_indices[:, 5]]  # 0: c->d, 1: d->c
        
        # a dominates b,c,d: e_ab==0, e_ac==0, e_ad==0
        a_doms = (e_ab == 0) & (e_ac == 0) & (e_ad == 0)
        # b dominates a,c,d: e_ab==1, e_bc==0, e_bd==0
        b_doms = (e_ab == 1) & (e_bc == 0) & (e_bd == 0)
        # c dominates a,b,d: e_ac==1, e_bc==1, e_cd==0
        c_doms = (e_ac == 1) & (e_bc == 1) & (e_cd == 0)
        # d dominates a,b,c: e_ad==1, e_bd==1, e_cd==1
        d_doms = (e_ad == 1) & (e_bd == 1) & (e_cd == 1)
        
        good = a_doms | b_doms | c_doms | d_doms
        bad_count = num_quads - np.sum(good)
        return bad_count
    
    # Genetic Algorithm Implementation
    def genetic_algorithm(n, time_limit):
        m = n * (n - 1) // 2
        start_search = time.time()
        
        pop_size = 50
        elite_count = 5
        mutation_rate = 0.02
        crossover_rate = 0.7
        
        # Initialize population
        population = np.random.randint(0, 2, size=(pop_size, m), dtype=np.int8)
        costs = np.array([evaluate_cost(population[i], n) for i in range(pop_size)])
        
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_orient = population[best_idx].copy()
        
        if best_cost == 0:
            return best_orient, True
        
        # Evolution loop
        generation = 0
        stagnant_generations = 0
        
        while time.time() - start_search < time_limit:
            generation += 1
            
            # Sort population by cost
            sort_indices = np.argsort(costs)
            population = population[sort_indices]
            costs = costs[sort_indices]
            
            # Check for improvement
            if costs[0] < best_cost:
                best_cost = costs[0]
                best_orient = population[0].copy()
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            
            if best_cost == 0:
                return best_orient, True
            
            # If stagnating for too long, restart with fresh random individuals
            if stagnant_generations > 20:
                num_fresh = pop_size // 2
                fresh_pop = np.random.randint(0, 2, size=(num_fresh, m), dtype=np.int8)
                fresh_costs = np.array([evaluate_cost(fresh_pop[i], n) for i in range(num_fresh)])
                
                # Keep best individuals from current population and add fresh ones
                keep_count = pop_size - num_fresh
                population = np.vstack([population[:keep_count], fresh_pop])
                costs = np.concatenate([costs[:keep_count], fresh_costs])
                stagnant_generations = 0
                
                # Update best
                best_idx = np.argmin(costs)
                if costs[best_idx] < best_cost:
                    best_cost = costs[best_idx]
                    best_orient = population[best_idx].copy()
                    if best_cost == 0:
                        return best_orient, True
            
            # Create next generation
            new_population = []
            
            # Elite selection: keep top performers
            for i in range(elite_count):
                new_population.append(population[i].copy())
            
            # Fill rest with offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def tournament_select():
                    idx1 = random.randint(0, pop_size - 1)
                    idx2 = random.randint(0, pop_size - 1)
                    if costs[idx1] <= costs[idx2]:
                        return idx1
                    else:
                        return idx2
                
                parent1_idx = tournament_select()
                parent2_idx = tournament_select()
                
                # Crossover
                if random.random() < crossover_rate:
                    # Uniform crossover
                    child = population[parent1_idx].copy()
                    mask = np.random.randint(0, 2, size=m, dtype=np.int8)
                    child[mask == 1] = population[parent2_idx][mask == 1]
                else:
                    # No crossover, just copy parent
                    child = population[parent1_idx].copy()
                
                # Mutation: bit flip
                if random.random() < 1.0:  # Always mutate
                    num_mutations = max(1, int(m * mutation_rate * random.uniform(0.5, 1.5)))
                    mut_positions = np.random.choice(m, size=num_mutations, replace=False)
                    child[mut_positions] = 1 - child[mut_positions]
                
                new_population.append(child)
            
            population = np.array(new_population, dtype=np.int8)
            costs = np.array([evaluate_cost(population[i], n) for i in range(pop_size)])
            
            # Update best
            best_idx = np.argmin(costs)
            if costs[best_idx] < best_cost:
                best_cost = costs[best_idx]
                best_orient = population[best_idx].copy()
                if best_cost == 0:
                    return best_orient, True
        
        return best_orient, (best_cost == 0)
    
    # Try to find a solution for n = 66
    if time.time() - start_time < deadline - 20:
        try:
            result_orient, success = genetic_algorithm(target_n, deadline - 20 - (time.time() - start_time))
            if success:
                # Convert numpy array to list
                orientation_list = result_orient.tolist()
                best_n = target_n
                best_orientation = orientation_list
        except Exception as e:
            pass  # Fall back to known solution
    
    return {'n': best_n, 'orientation': best_orientation}

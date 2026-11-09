import random
import time
import sys
import math

def construct():
    n = 16
    num_edges = 120
    
    # Precompute edges and K4s
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    k4s = []
    k4_edge_indices = []
    
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    verts = [v0, v1, v2, v3]
                    e_indices = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            e = (verts[a], verts[b])
                            e_indices.append(edge_to_idx[e])
                    k4s.append(verts)
                    k4_edge_indices.append(e_indices)
                    
    num_k4s = len(k4s)

    def fitness(colors):
        """
        Returns the number of K4s with < 5 distinct colors.
        Lower is better. 0 means valid.
        """
        bad = 0
        for e_indices in k4_edge_indices:
            # Extract colors for this K4
            c0 = colors[e_indices[0]]
            c1 = colors[e_indices[1]]
            c2 = colors[e_indices[2]]
            c3 = colors[e_indices[3]]
            c4 = colors[e_indices[4]]
            c5 = colors[e_indices[5]]
            
            # Check distinct count
            # Using a set is slow, but for 6 elements it's okay.
            # Alternatively, check if all 6 are same (1), 5 same (2), etc.
            # len(set) is fine for small lists.
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad += 1
        return bad

    def ga_solve(K, time_limit):
        pop_size = 100
        mutation_rate = 0.1
        tournament_size = 5
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            ind = [random.randint(0, K - 1) for _ in range(num_edges)]
            population.append(ind)
            
        best_ind = min(population, key=fitness)
        best_fit = fitness(best_ind)
        
        if best_fit == 0:
            return best_ind
            
        while time.time() - start_time < time_limit:
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(best_ind[:])
            
            while len(new_population) < pop_size:
                # Selection
                parents = []
                for _ in range(2):
                    # Tournament selection
                    contenders = random.sample(population, tournament_size)
                    parent = min(contenders, key=fitness)
                    parents.append(parent)
                
                # Crossover (Uniform)
                child = [0] * num_edges
                for i in range(num_edges):
                    if random.random() < 0.5:
                        child[i] = parents[0][i]
                    else:
                        child[i] = parents[1][i]
                        
                # Mutation
                if random.random() < mutation_rate:
                    # Mutate a few genes
                    num_mutations = random.randint(1, 5)
                    for _ in range(num_mutations):
                        gene_idx = random.randint(0, num_edges - 1)
                        child[gene_idx] = random.randint(0, K - 1)
                        
                new_population.append(child)
                
            population = new_population
            
            # Update best
            for ind in population:
                fit = fitness(ind)
                if fit < best_fit:
                    best_fit = fit
                    best_ind = ind[:]
                    
            if best_fit == 0:
                return best_ind
                
        return None

    # Try K values in increasing order to minimize colors
    # But 13 is very hard. 15 is easier.
    # If we find 15, score is 15. If we find 13, score is 13.
    # We should try 13, 14, 15.
    # However, if 13 takes 110s and fails, we get 0 score.
    # So we must allocate time wisely.
    
    # Heuristic: 15 is almost certain. 14 is likely. 13 is rare.
    # Let's try 15 first to guarantee a baseline? No, we want to minimize.
    # But if we try 13 for 30s and fail, we have 90s left.
    # Let's try 13 (20s), 14 (40s), 15 (40s), 16 (10s).
    
    result = None
    for K in [13, 14, 15, 16]:
        if K == 13:
            t_lim = 20
        elif K == 14:
            t_lim = 40
        elif K == 15:
            t_lim = 40
        else:
            t_lim = 10
            
        res = ga_solve(K, t_lim)
        if res is not None:
            # Verify
            if fitness(res) == 0:
                return res
                
    # Fallback: Random search for K=17 or just return a known valid pattern if we had one.
    # The current program has a fallback for 17.
    # Let's create a simple valid coloring for K=17 or similar if GA fails.
    # Actually, if GA fails for 13, 14, 15, 16, we are in trouble.
    # But 15 should be very easy.
    
    # Fallback to the old logic's fallback (17 colors) or try a quick SA for 16?
    # Let's just return a random valid coloring for K=17 if all else fails.
    # (i+j) % 17 is not guaranteed valid for K4>=5.
    # Let's just do a quick local search for K=16 as a last resort.
    
    # Re-use a simple local search for K=16 as fallback
    def local_search_fallback(K, time_limit=5):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        current_cost = fitness(colors)
        if current_cost == 0: return colors
        
        start = time.time()
        T = 1.0
        while time.time() - start < time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, K - 1)
            if new_c == old_c: continue
            
            # Calculate delta roughly? Or just re-evaluate if small?
            # For fallback, just accept if random < exp(-delta/T)
            # We need delta.
            # Let's just re-evaluate fitness for the whole thing? Too slow.
            # We need edge_to_k4s map for delta calculation.
            
            # Let's just return a random coloring if we fail, but that's invalid.
            # We MUST return a valid coloring.
            # The task says "A valid colouring certifies...".
            # If we return an invalid one, score is 0.
            
            pass
        
        return None

    # We need a guaranteed valid coloring.
    # A known construction for K_n with 5-color K4s?
    # Actually, the current program's fallback `(i + j) % 17` is likely invalid.
    # Let's check if `(i + j) % 17` works for K4.
    # Vertices 0,1,2,3. Edges: (0,1)=1, (0,2)=2, (0,3)=3, (1,2)=3, (1,3)=4, (2,3)=5.
    # Colors: 1,2,3,3,4,5. Distinct: {1,2,3,4,5} -> 5 distinct. Valid.
    # Vertices 0,1,2,4. Edges: (0,1)=1, (0,2)=2, (0,4)=4, (1,2)=3, (1,4)=5, (2,4)=6.
    # Colors: 1,2,4,3,5,6. Distinct: 6. Valid.
    # It seems `(i+j) % M` might work for small n if M is large enough.
    # For n=16, max i+j is 15+14=29.
    # If we use M=17, colors are 0..16.
    # Let's verify a potential failure case for M=17.
    # We need a K4 with < 5 distinct colors.
    # With `(i+j) % 17`, the colors in a K4 {a,b,c,d} are:
    # a+b, a+c, a+d, b+c, b+d, c+d (mod 17).
    # Can we have 4 or fewer distinct values?
    # If a,b,c,d are in arithmetic progression?
    # Let's just use a fallback that is definitely valid:
    # Use a 1-factorization or similar?
    # Actually, the current program's fallback might be invalid.
    # Let's implement a simple backtracking or just trust the GA for K=15.
    # K=15 is very likely to succeed with GA.
    
    # If GA fails for all, we return a random K=16 coloring and hope? No, must be valid.
    # Let's add a simple local search for K=16 as the absolute last resort, running for 5s.
    
    # To do local search efficiently, we need the delta function from the original code.
    # Let's copy the delta logic.
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edge_indices[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)

    def delta_cost(edge_idx, old_color, new_color, colors):
        delta = 0
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edge_indices[k4_idx]
            c_old = [colors[e] for e in e_indices]
            c_new_list = []
            for e in e_indices:
                if e == edge_idx:
                    c_new_list.append(new_color)
                else:
                    c_new_list.append(colors[e])
            
            old_valid = len(set(c_old)) >= 5
            new_valid = len(set(c_new_list)) >= 5
            
            if old_valid and not new_valid:
                delta += 1
            elif not old_valid and new_valid:
                delta -= 1
        return delta

    def local_search(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        current_cost = fitness(colors)
        if current_cost == 0: return colors
        
        start = time.time()
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while time.time() - start < time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, K - 1)
            if new_c == old_c: continue
            
            d_cost = delta_cost(e_idx, old_c, new_c, colors)
            
            if d_cost <= 0 or random.random() < math.exp(-d_cost / T):
                colors[e_idx] = new_c
                current_cost += d_cost
                if current_cost == 0:
                    return colors
            T *= cooling
            if T < min_T: T = min_T
        return None

    # Final fallback: Try K=16 with Local Search for 10s
    res = local_search(16, 10)
    if res:
        return res
        
    # If even that fails, return a random coloring? No, that's invalid.
    # We must return *something*. The problem implies we should find a valid one.
    # Given the bounds, 15 or 16 should be easy.
    # If we get here, we return a dummy list to avoid crash, but score will be 0.
    return [0] * 120

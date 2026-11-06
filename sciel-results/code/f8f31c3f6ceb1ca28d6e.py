import time
import random
import sys
from itertools import combinations

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    Strategy:
    1. Base: 8 disjoint K4s (n=32, alpha=8, C5-free).
    2. Try to extend to n=33, 34, 35 using complement-graph reformulation
       and a genetic algorithm optimizing edge density under constraints.
    3. Use fast C5 and IS checks with bitmasks.
    4. Fall back to n=32 if extension fails.
    """
    
    def has_c5_fast(n, adj_mask):
        """Fast C5 detection using bitmask operations."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)):
                    continue
                w_mask = adj_mask[v] & ~(1 << u) & ~(1 << v)
                if not w_mask:
                    continue
                for w in range(n):
                    if not (w_mask & (1 << w)):
                        continue
                    x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if not x_mask:
                        continue
                    for x in range(n):
                        if not (x_mask & (1 << x)):
                            continue
                        y_mask = adj_mask[x] & adj_mask[u] & ~(1 << v) & ~(1 << w) & ~(1 << x)
                        if y_mask:
                            return True
        return False
    
    def has_is_k_fast(n, adj_mask, k):
        """Check for independent set of size k using bitmask DFS with pruning."""
        
        def greedy_is_size(cand):
            """Greedy upper bound on independent set size in cand."""
            count = 0
            c = cand
            while c:
                v = -1
                min_deg = n + 1
                cc = c
                while cc:
                    lsb = cc & -cc
                    vi = lsb.bit_length() - 1
                    deg = (adj_mask[vi] & c).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        v = vi
                    cc -= lsb
                if v == -1:
                    break
                count += 1
                c &= ~adj_mask[v] & ~(1 << v)
            return count
        
        def dfs(cand, rem):
            if rem == 0:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < rem:
                return False
            if greedy_is_size(cand) < rem:
                return False
            
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1:
                return False
                
            lsb_v = 1 << v
            
            new_cand_incl = cand & ~adj_mask[v] & ~lsb_v
            if new_cand_incl.bit_count() >= rem - 1:
                if dfs(new_cand_incl, rem - 1):
                    return True
            
            new_cand_excl = cand & ~lsb_v
            if new_cand_excl.bit_count() >= rem:
                if dfs(new_cand_excl, rem):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, k)
    
    def is_valid(n, adj_mask):
        if has_c5_fast(n, adj_mask):
            return False
        if has_is_k_fast(n, adj_mask, 9):
            return False
        return True
    
    def adj_to_edges(n, adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges
    
    def get_base_32():
        n = 32
        adj_mask = [0] * n
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        return n, adj_mask
    
    # Complement graph reformulation:
    # Original: G is C5-free and alpha(G) < 9
    # Complement G': G' has no K9 (since alpha(G) = omega(G')) and no induced C5
    # (since C5 is self-complementary, C5-free in G means C5-free in G')
    # Wait, that's not quite right. Let me reconsider.
    # 
    # Actually, the complement reformulation is:
    # - G is C5-free <=> G' is C5-free (since C5 is self-complementary)
    # - alpha(G) < 9 <=> omega(G') < 9 <=> G' is K9-free
    #
    # So we want G' to be C5-free AND K9-free.
    # This is a different constraint set but equivalent.
    
    def has_k9_fast(n, adj_mask):
        """Check for K9 in adj_mask using bitmask."""
        # A K9 means 9 vertices all mutually adjacent.
        # Use Bron-Kerbosch or simple enumeration with pruning.
        
        def dfs(cand, size):
            if size == 9:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < 9 - size:
                return False
            
            # Pick vertex with max degree in cand
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1:
                return False
                
            lsb_v = 1 << v
            
            # Include v
            new_cand = cand & adj_mask[v] & ~lsb_v
            if new_cand.bit_count() >= 9 - (size + 1):
                if dfs(new_cand, size + 1):
                    return True
            
            # Exclude v
            new_cand_excl = cand & ~lsb_v
            if new_cand_excl.bit_count() >= 9 - size:
                if dfs(new_cand_excl, size):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, 0)
    
    def is_valid_complement(n, comp_adj_mask):
        """Check if complement graph is C5-free and K9-free."""
        if has_c5_fast(n, comp_adj_mask):
            return False
        if has_k9_fast(n, comp_adj_mask):
            return False
        return True
    
    def complement_adj_mask(n, adj_mask):
        """Compute complement adjacency mask."""
        comp = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)):
                    comp[u] |= (1 << v)
                    comp[v] |= (1 << u)
        return comp
    
    def edge_count(n, adj_mask):
        count = 0
        for u in range(n):
            count += adj_mask[u].bit_count()
        return count // 2
    
    # Genetic algorithm on complement graph
    def ga_search_complement(n, time_budget, init_adj=None):
        """GA searching for complement graph that is C5-free and K9-free."""
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Population of complement adjacency masks
        pop_size = 40
        population = []
        
        if init_adj is not None:
            population.append(init_adj[:])
        
        while len(population) < pop_size:
            # Random initialization biased toward dense graphs
            # (dense complement = sparse original, which helps with C5-free)
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            population.append(adj)
        
        # Fitness: maximize edges while being valid
        # Use a smooth surrogate: fitness = edge_count - penalty
        def fitness(adj):
            ec = edge_count(n, adj)
            # Check C5
            if has_c5_fast(n, adj):
                # Penalty for C5
                return ec - 1000
            # Check K9
            if has_k9_fast(n, adj):
                return ec - 500
            return ec
        
        best_adj = None
        best_fit = -1
        
        gen = 0
        while time.time() < deadline:
            gen += 1
            
            # Evaluate fitness
            fits = [fitness(adj) for adj in population]
            
            for i in range(len(population)):
                if fits[i] > best_fit:
                    best_fit = fits[i]
                    best_adj = population[i][:]
            
            # Check if we found a valid solution
            if best_adj is not None and is_valid_complement(n, best_adj):
                return best_adj
            
            # Selection: tournament selection
            def tournament_select():
                idxs = random.sample(range(len(population)), min(3, len(population)))
                best_idx = max(idxs, key=lambda i: fits[i])
                return population[best_idx][:]
            
            # Crossover: uniform crossover on edge presence
            def crossover(a, b):
                child = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            if a[u] & (1 << v):
                                child[u] |= (1 << v)
                                child[v] |= (1 << u)
                        else:
                            if b[u] & (1 << v):
                                child[u] |= (1 << v)
                                child[v] |= (1 << u)
                return child
            
            # Mutation: flip random edges
            def mutate(adj, rate=0.05):
                adj = adj[:]
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < rate:
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                return adj
            
            # Create next generation
            new_pop = []
            # Elitism: keep top 2
            top_idxs = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
            for i in top_idxs[:2]:
                new_pop.append(population[i][:])
            
            while len(new_pop) < pop_size:
                parent1 = tournament_select()
                parent2 = tournament_select()
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_pop.append(child)
            
            population = new_pop
    
        return best_adj
    
    # Main logic with time management
    total_time = 145.0
    start_total = time.time()
    
    # Try n=35, 34, 33 in descending order
    for target_n in [35, 34, 33]:
        if time.time() - start_total > total_time - 10:
            break
        
        time_budget = min(30.0, total_time - (time.time() - start_total) - 5)
        if time_budget < 5:
            break
        
        # Initialize from base 32 + random extra vertices
        base_n, base_adj = get_base_32()
        init_comp = None
        
        if target_n > 32:
            # Start with complement of base 32 (which is 8 disjoint independent sets of size 4)
            # Plus random edges for extra vertices
            comp = [0] * target_n
            # Base 32 complement: no edges within K4s, all edges between K4s
            for b1 in range(8):
                for b2 in range(b1 + 1, 8):
                    for i in range(4):
                        for j in range(4):
                            v1 = b1 * 4 + i
                            v2 = b2 * 4 + j
                            comp[v1] |= (1 << v2)
                            comp[v2] |= (1 << v1)
            # Extra vertices: random connections
            for v in range(32, target_n):
                for u in range(v):
                    if random.random() < 0.5:
                        comp[v] |= (1 << u)
                        comp[u] |= (1 << v)
            init_comp = comp
        
        result = ga_search_complement(target_n, time_budget, init_adj=init_comp)
        
        if result is not None and is_valid_complement(target_n, result):
            # Convert complement back to original
            orig_adj = complement_adj_mask(target_n, result)
            if is_valid(target_n, orig_adj):
                edges = adj_to_edges(target_n, orig_adj)
                return {'n': target_n, 'edges': edges}
    
    # Fallback to verified n=32
    n, adj_mask = get_base_32()
    edges = adj_to_edges(n, adj_mask)
    return {'n': n, 'edges': edges}

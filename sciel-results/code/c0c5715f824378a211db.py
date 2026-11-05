import time
import random

_START = time.time()
_DEADLINE = 143  # safety margin for 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (adj[v] & candidates).bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        if best_v == -1:
            return None
        candidates &= ~(1 << best_v)
        new_candidates = candidates & ~adj[best_v]
        res = backtrack(new_candidates, current_set + [best_v])
        if res:
            return res
        res = backtrack(candidates, current_set)
        if res:
            return res
        return None
    return backtrack((1 << n) - 1, [])

def _count_violations(adj, n):
    """Count K4s and independent 6-sets (capped)."""
    k4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        k4_count += 1
                        if k4_count > 100:
                            return k4_count, 0
    ind6_count = 0
    # Quick check: try to find one ind6
    if _has_ind_6(adj, n):
        ind6_count = 1
    return k4_count, ind6_count

def _is_valid(adj, n):
    if _has_k4(adj, n):
        return False
    if _has_ind_6(adj, n):
        return False
    return True

def _paley_adj(n):
    """Return adjacency list for Paley graph of order n (n must be prime ≡ 1 mod 4)."""
    if n <= 1:
        return [0] * n
    p = n
    # Check if p is prime
    def is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True
    if not is_prime(p) or p % 4 != 1:
        # Fallback: use a random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, time_limit):
    """Targeted local search: fix K4s and Ind6s iteratively."""
    start = time.time()
    while time.time() - start < time_limit:
        k4 = _has_k4(adj, n)
        if k4:
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        ind6 = _has_ind_6(adj, n)
        if ind6:
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        return True
    return False

def _genetic_search(n, time_limit, pop_size=60, elite_size=10):
    """Genetic algorithm seeded with Paley-like structures."""
    start = time.time()
    
    # Initialize population with diverse seeds
    population = []
    # Seed 1: Paley-like (use n=17 Paley extended randomly)
    base_adj = _paley_adj(17)
    for _ in range(pop_size // 4):
        adj = [0] * n
        for i in range(min(17, n)):
            adj[i] = base_adj[i]
        for i in range(17, n):
            for j in range(i):
                if random.random() < 0.45:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Random perturbation
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.05:
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        population.append(adj)
    
    # Seed 2: Random graphs with various densities
    for density in [0.35, 0.4, 0.45, 0.5]:
        for _ in range(pop_size // 8):
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            population.append(adj)
    
    # Seed 3: Complements of random graphs
    for _ in range(pop_size // 8):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.55:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        population.append(adj)
    
    def fitness(adj):
        k4s, ind6s = _count_violations(adj, n)
        # Lower is better
        return k4s * 2 + ind6s * 5  # Ind6 is harder to fix
    
    # Sort and keep best
    population.sort(key=fitness)
    
    generations = 0
    while time.time() - start < time_limit:
        generations += 1
        
        # Check if any is valid
        for adj in population[:10]:
            if _is_valid(adj, n):
                return adj
        
        # Selection: keep elites
        new_pop = [list(a) for a in population[:elite_size]]
        
        # Crossover and mutation
        while len(new_pop) < pop_size:
            if time.time() - start > time_limit:
                break
            # Tournament selection
            a_idx = random.choice(range(min(20, len(population))))
            b_idx = random.choice(range(min(20, len(population))))
            a = population[a_idx]
            b = population[b_idx]
            
            # Uniform crossover
            child = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        if a[i] & (1 << j):
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
                    else:
                        if b[i] & (1 << j):
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
            
            # Mutation
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.03:
                        if child[i] & (1 << j):
                            child[i] &= ~(1 << j)
                            child[j] &= ~(1 << i)
                        else:
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
            
            new_pop.append(child)
        
        # Local repair on some individuals
        for idx in range(min(5, len(new_pop))):
            if time.time() - start > time_limit:
                break
            if _local_search(new_pop[idx], n, 0.05):
                pass  # valid
        
        population = new_pop
        population.sort(key=fitness)
    
    # Final check
    for adj in population[:20]:
        if _is_valid(adj, n):
            return adj
    return None

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    # Baseline: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Strategy: Try n from 18 upward, with genetic algorithm for larger n
    # For n <= 20, local search from random init is fast
    # For n > 20, use genetic algorithm seeded with Paley structure
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 3:
            break
        
        # Time allocation: more time for larger n
        remaining = _DEADLINE - _elapsed() - 3
        steps_left = 36 - n
        time_slice = remaining / max(1, steps_left)
        time_slice = min(time_slice, 15.0)
        if time_slice < 0.5:
            time_slice = 0.5
        
        found = False
        
        # For small n, try local search first (faster)
        if n <= 22:
            for attempt in range(2):
                if _elapsed() > _DEADLINE - 3:
                    break
                alloc = time_slice / 2
                adj_init = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.45:
                            adj_init[i] |= (1 << j)
                            adj_init[j] |= (1 << i)
                if _local_search(adj_init, n, alloc):
                    best_n = n
                    best_edges = _adj_to_edges(adj_init, n)
                    found = True
                    break
        
        if found:
            continue
        
        # Genetic algorithm for all n (especially effective for larger n)
        if _elapsed() < _DEADLINE - 3:
            alloc = min(time_slice, _DEADLINE - 3 - _elapsed())
            result_adj = _genetic_search(n, alloc)
            if result_adj is not None:
                best_n = n
                best_edges = _adj_to_edges(result_adj, n)
                continue
        
        # Additional local search attempts with different seeds
        if _elapsed() < _DEADLINE - 2:
            for seed_val in [42, 123]:
                if _elapsed() > _DEADLINE - 2:
                    break
                random.seed(seed_val)
                adj_init = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.42:
                            adj_init[i] |= (1 << j)
                            adj_init[j] |= (1 << i)
                if _local_search(adj_init, n, 1.0):
                    best_n = n
                    best_edges = _adj_to_edges(adj_init, n)
                    break
    
    return {'n': best_n, 'edges': best_edges}

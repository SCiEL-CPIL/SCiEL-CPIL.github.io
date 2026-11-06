import time
import random
import itertools

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def get_edge_index(i, j, n):
        """Map (i, j) with i < j to index in the flat array."""
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def verify(n, colors):
        """
        Verify that no color class contains a C6.
        Returns True if valid, False otherwise.
        """
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u:
                        continue
                    for w in adj_list[v]:
                        if w == u:
                            continue
                        for x in adj_list[w]:
                            if x == u or x == v:
                                continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w:
                                    continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x:
                                        continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def count_c6(n, colors):
        """
        Count the number of monochromatic C6s.
        """
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        count = 0
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u:
                        continue
                    for w in adj_list[v]:
                        if w == u:
                            continue
                        for x in adj_list[w]:
                            if x == u or x == v:
                                continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w:
                                    continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x:
                                        continue
                                    if (adj[c][z] >> u) & 1:
                                        count += 1
        return count

    def genetic_search(n, time_limit):
        """
        Genetic Algorithm to minimize monochromatic C6s.
        """
        num_edges = n * (n - 1) // 2
        pop_size = 20
        population = []
        
        # Initialize population
        for _ in range(pop_size):
            population.append([random.randint(0, 3) for _ in range(num_edges)])
            
        # Evaluate initial population
        scores = [count_c6(n, ind) for ind in population]
        
        deadline = time.time() + time_limit
        best_score = min(scores)
        best_ind = population[scores.index(best_score)]
        
        while time.time() < deadline:
            # Selection
            new_population = []
            
            # Elitism: keep the best 2
            sorted_inds = sorted(zip(scores, population), key=lambda x: x[0])
            for s, ind in sorted_inds[:2]:
                new_population.append(ind)
                
            # Fill rest with offspring
            while len(new_population) < pop_size:
                # Tournament selection
                p1 = random.sample(range(pop_size), 2)
                p1 = population[p1[0] if scores[p1[0]] < scores[p1[1]] else p1[1]]
                
                p2 = random.sample(range(pop_size), 2)
                p2 = population[p2[0] if scores[p2[0]] < scores[p2[1]] else p2[1]]
                
                # Uniform crossover
                child = [random.choice([p1[i], p2[i]]) for i in range(num_edges)]
                
                # Mutation
                # Mutate a few edges
                for _ in range(3):
                    idx = random.randint(0, num_edges - 1)
                    child[idx] = random.randint(0, 3)
                    
                new_population.append(child)
                
            population = new_population
            scores = [count_c6(n, ind) for ind in population]
            
            current_best_score = min(scores)
            if current_best_score < best_score:
                best_score = current_best_score
                best_ind = population[scores.index(best_score)]
                
            if best_score == 0:
                return best_ind

        return best_ind if best_score == 0 else None

    # Strategy:
    # 1. Try n=17 with GA.
    # 2. Try n=16 with GA.
    # 3. Fallback to repair_search for smaller n.

    if time.time() < hard_deadline - 10:
        res = genetic_search(17, 50)
        if res:
            if verify(17, res):
                return {'n': 17, 'colors': res}

    if time.time() < hard_deadline - 10:
        res = genetic_search(16, 30)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    if time.time() < hard_deadline - 5:
        res = genetic_search(15, 15)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    if time.time() < hard_deadline - 5:
        res = genetic_search(14, 10)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}

    if time.time() < hard_deadline - 5:
        res = genetic_search(13, 5)
        if res:
            if verify(13, res):
                return {'n': 13, 'colors': res}

    if time.time() < hard_deadline - 2:
        res = genetic_search(12, 2)
        if res:
            if verify(12, res):
                return {'n': 12, 'colors': res}

    # Fallback to n=10
    if time.time() < hard_deadline - 2:
        res = genetic_search(10, 2)
        if res:
            if verify(10, res):
                return {'n': 10, 'colors': res}

    # Absolute fallback: n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m_total = len(all_edges)
    
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    
    def has_q3_with_edge(adj, u, v):
        # Check if the graph (with edge u-v already added) contains a Q3 containing (u,v)
        # This is an incremental check assuming (u,v) is the "new" edge.
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
            
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                
                for y1 in y1_candidates:
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                            
                            for y2 in y2_candidates:
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 in (z1, z2): continue
                                if w2 in (z1, z2): continue
                                
                                if not ((adj[y1] >> y2) & 1): continue
                                
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
                                    
        return False

    def get_shrikhande_edges():
        shrikhande_edges = []
        for i in range(4):
            for j in range(4):
                u = i * 4 + j
                neighbors = [
                    ((i + 1) % 4) * 4 + j,
                    ((i - 1) % 4) * 4 + j,
                    i * 4 + ((j + 1) % 4),
                    i * 4 + ((j - 1) % 4),
                    ((i + 1) % 4) * 4 + ((j + 1) % 4),
                    ((i - 1) % 4) * 4 + ((j - 1) % 4)
                ]
                for v in neighbors:
                    if v > u:
                        shrikhande_edges.append((u, v))
        return shrikhande_edges

    def run_search(seed, initial_edges=None):
        random.seed(seed)
        adj = [0] * n
        edge_set = set()
        
        if initial_edges is not None:
            for u, v in initial_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        else:
            # 1. Greedy Construction
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # 2. Local Search (Hill Climbing)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for u, v in all_edges:
                if (u, v) not in edge_set:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not has_q3_with_edge(adj, u, v):
                        edge_set.add((u, v))
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            if not improved and len(edge_set) > 0:
                u, v = random.choice(list(edge_set))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.remove((u, v))
                
                added = False
                for u2, v2 in all_edges:
                    if (u2, v2) not in edge_set and (u2, v2) != (u, v):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        if not has_q3_with_edge(adj, u2, v2):
                            edge_set.add((u2, v2))
                            added = True
                            break
                        else:
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                
                if not added:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                else:
                    improved = True
        
        return len(edge_set), list(edge_set), adj

    def repair(edge_set):
        """Remove edges to eliminate any Q3 subgraphs while preserving maximum density."""
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        removed = 0
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for u, v in list(edge_set):
                # Check if removing this edge is needed
                # Try adding it back temporarily to see if it creates a Q3
                if not has_q3_with_edge(adj, u, v):
                    continue
                # This edge might be part of a Q3, try removing it
                edge_set.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                removed += 1
                progress = True
        
        return edge_set, adj, removed

    def genetic_crossover(parent1_edges, parent2_edges):
        """Custom crossover that combines edge sets from two high-quality parents."""
        set1 = set(parent1_edges)
        set2 = set(parent2_edges)
        
        # Union with random selection to create diversity
        child_edges = set()
        for e in all_edges:
            in1 = e in set1
            in2 = e in set2
            if in1 and in2:
                child_edges.add(e)
            elif in1 or in2:
                if random.random() < 0.7:  # Prefer keeping edges from at least one parent
                    child_edges.add(e)
            else:
                if random.random() < 0.3:  # Occasionally add new edges
                    child_edges.add(e)
        
        return child_edges

    def run_genetic_algorithm(initial_pop, pop_size=10, generations=20):
        """Genetic algorithm with custom crossover and repair."""
        # Initialize population
        population = []
        for i in range(pop_size):
            if i < len(initial_pop):
                count, edges, adj = initial_pop[i]
                population.append((count, edges, adj))
            else:
                count, edges, adj = run_search(i + 1000)
                population.append((count, edges, adj))
        
        best = max(population, key=lambda x: x[0])
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Sort by fitness
            population.sort(key=lambda x: x[0], reverse=True)
            
            new_population = []
            
            # Elitism: keep top 2
            for i in range(min(2, len(population))):
                new_population.append(population[i])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def tournament():
                    candidates = random.sample(population, min(3, len(population)))
                    return max(candidates, key=lambda x: x[0])
                
                p1 = tournament()
                p2 = tournament()
                
                # Crossover
                child_edges = genetic_crossover(p1[1], p2[1])
                
                # Repair
                child_edges, child_adj, _ = repair(child_edges)
                
                # Local improvement
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for u, v in all_edges:
                        if (u, v) not in child_edges:
                            child_adj[u] |= (1 << v)
                            child_adj[v] |= (1 << u)
                            if not has_q3_with_edge(child_adj, u, v):
                                child_edges.add((u, v))
                                improved = True
                            else:
                                child_adj[u] &= ~(1 << v)
                                child_adj[v] &= ~(1 << u)
                
                new_population.append((len(child_edges), list(child_edges), child_adj))
            
            population = new_population
            best = max(population, key=lambda x: x[0])
        
        return best

    # Run multiple restarts
    restart = 0
    
    # Try Shrikhande graph as seed
    shrikhande = get_shrikhande_edges()
    count, edges_list, adj = run_search(100, shrikhande)
    if count > best_count:
        best_count = count
        best_edges = edges_list
        best_adj = adj[:]
        
    # Try random starts
    while time.time() < deadline - 5:
        count, edges_list, adj = run_search(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
            best_adj = adj[:]
        restart += 1

    if best_count == 0:
        random.seed(0)
        edges = all_edges[:]
        random.shuffle(edges)
        adj = [0] * n
        edge_set = set()
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3_with_edge(adj, u, v):
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        best_count = len(edge_set)
        best_edges = list(edge_set)

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())

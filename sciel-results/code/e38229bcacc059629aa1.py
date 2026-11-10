import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    n = 20
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Map pair (i, j) to an index
    pair_to_idx = {}
    idx_to_pair = [None] * num_pairs
    for idx, (i, j) in enumerate(all_pairs):
        pair_to_idx[(i, j)] = idx
        idx_to_pair[idx] = (i, j)

    # Precompute triples for K3,3 checking
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute for each triple, the indices of the vertices
    # And precompute masks for each vertex
    vertex_masks = [1 << i for i in range(n)]
    
    def popcount(x):
        return x.bit_count()

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    def is_k33_free(adj_mask):
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def count_violations(adj_mask):
        """Count number of triples with >2 common neighbors."""
        violations = 0
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                violations += 1
        return violations

    def can_add_fast(u, v, adj_mask):
        """
        Fast check: only triples containing u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        for a, b, c in triples:
            if a != u and a != v and b != u and b != v and c != u and c != v:
                continue
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        return True

    def greedy_add(adj_mask):
        """Add all possible edges greedily in a specific order."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add_fast(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            changed = True
        return adj_mask

    def random_greedy_add(adj_mask, rng):
        """Add edges in random order."""
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        for i, j in pairs:
            if not (adj_mask[i] & (1 << j)):
                if can_add_fast(i, j, adj_mask):
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def smart_repair(adj_mask):
        """Remove edges to eliminate all K3,3 violations."""
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Find a violation
            viol = None
            for u, v, w in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    # Get list of common neighbors
                    common_list = [x for x in range(n) if common & (1 << x)]
                    # Remove edge from u to first common neighbor
                    c = common_list[0]
                    viol = (u, c)
                    break
            
            if viol is None:
                break
            
            u, c = viol
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
        
        return adj_mask

    def local_search_remove_readd(adj_mask, max_steps=50, rng=None):
        """Try removing one edge and greedily re-adding to potentially gain edges."""
        if rng is None:
            rng = random.Random(42)
        
        best_adj = list(adj_mask)
        best_cnt = edge_count(best_adj)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Get all edges
            current_edges = []
            for i in range(n):
                m = best_adj[i]
                for j in range(i + 1, n):
                    if m & (1 << j):
                        current_edges.append((i, j))
            
            if not current_edges:
                break
            
            # Pick a random edge to remove
            u, v = rng.choice(current_edges)
            
            new_adj = list(best_adj)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Greedy add
            new_adj = greedy_add(new_adj)
            cnt = edge_count(new_adj)
            
            if cnt > best_cnt:
                best_cnt = cnt
                best_adj = new_adj
            elif cnt == best_cnt and rng.random() < 0.1:
                # Occasional move to same-count state to escape local optima
                best_adj = new_adj
        
        return best_adj, best_cnt

    def multi_point_local_search(adj_mask, max_steps=30, rng=None):
        """Try removing 2 edges and greedily re-adding."""
        if rng is None:
            rng = random.Random(42)
        
        best_adj = list(adj_mask)
        best_cnt = edge_count(best_adj)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            current_edges = []
            for i in range(n):
                m = best_adj[i]
                for j in range(i + 1, n):
                    if m & (1 << j):
                        current_edges.append((i, j))
            
            if len(current_edges) < 2:
                break
            
            # Pick 2 random edges to remove
            idx1 = rng.randint(0, len(current_edges) - 1)
            idx2 = rng.randint(0, len(current_edges) - 1)
            if idx1 == idx2:
                continue
            
            e1 = current_edges[idx1]
            e2 = current_edges[idx2]
            
            new_adj = list(best_adj)
            u1, v1 = e1
            u2, v2 = e2
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] &= ~(1 << v2)
            new_adj[v2] &= ~(1 << u2)
            
            # Greedy add
            new_adj = greedy_add(new_adj)
            cnt = edge_count(new_adj)
            
            if cnt > best_cnt:
                best_cnt = cnt
                best_adj = new_adj
        
        return best_adj, best_cnt

    def genetic_algorithm(pop_size=40, generations=100, rng=None):
        """Genetic algorithm with crossover and mutation."""
        if rng is None:
            rng = random.Random(42)
        
        # Initialize population with diverse random graphs
        population = []
        for i in range(pop_size):
            adj = [0] * n
            density = 0.5 + 0.4 * (i / pop_size)
            for x in range(n):
                for y in range(x + 1, n):
                    if rng.random() < density:
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
            adj = smart_repair(adj)
            adj = greedy_add(adj)
            population.append(adj)
        
        best_adj = population[0]
        best_cnt = edge_count(best_adj)
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Evaluate fitness
            fitnesses = []
            for adj in population:
                cnt = edge_count(adj)
                viol = count_violations(adj)
                fitness = cnt * 10 - viol * 100
                fitnesses.append(fitness)
                if cnt > best_cnt and viol == 0:
                    best_cnt = cnt
                    best_adj = list(adj)
            
            # Selection: keep top half
            sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
            parents = [population[i] for i in sorted_idx[:pop_size // 2]]
            
            # Create new population
            new_population = []
            # Elitism: keep best
            new_population.append(list(best_adj))
            
            while len(new_population) < pop_size:
                if time.time() > deadline:
                    break
                
                # Crossover
                p1 = rng.choice(parents)
                p2 = rng.choice(parents)
                
                child = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        # Randomly choose from parent1 or parent2
                        if rng.random() < 0.5:
                            if p1[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if p2[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                
                # Mutation: flip 1-3 edges
                num_flips = rng.randint(1, 3)
                for _ in range(num_flips):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u != v:
                        if child[u] & (1 << v):
                            child[u] &= ~(1 << v)
                            child[v] &= ~(1 << u)
                        else:
                            child[u] |= (1 << v)
                            child[v] |= (1 << u)
                
                # Repair and optimize
                child = smart_repair(child)
                child = greedy_add(child)
                new_population.append(child)
            
            population = new_population
        
        return best_adj, best_cnt

    def structured_construction(rng):
        """Create a structured dense graph using algebraic methods."""
        adj = [0] * n
        
        # Try a circulant-like structure
        # For n=20, try connections based on differences
        # K3,3-free means no 3 vertices with 3 common neighbors
        
        # Start with a moderately dense random graph
        density = rng.uniform(0.6, 0.85)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj

    best_adj = None
    best_edges = -1
    
    # Strategy 1: Multiple random restarts with local search
    for seed in range(60):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(seed)
        adj = structured_construction(rng)
        adj = smart_repair(adj)
        adj = greedy_add(adj)
        
        cnt = edge_count(adj)
        if cnt > best_edges and is_k33_free(adj):
            best_edges = cnt
            best_adj = list(adj)
        
        # Local search
        if time.time() < deadline - 15:
            adj, cnt = local_search_remove_readd(adj, max_steps=30, rng=rng)
            if cnt > best_edges and is_k33_free(adj):
                best_edges = cnt
                best_adj = list(adj)
    
    # Strategy 2: Multi-point local search from best
    if best_adj is not None and time.time() < deadline - 10:
        rng = random.Random(12345)
        adj, cnt = multi_point_local_search(best_adj, max_steps=40, rng=rng)
        if cnt > best_edges and is_k33_free(adj):
            best_edges = cnt
            best_adj = list(adj)
    
    # Strategy 3: Genetic algorithm
    if time.time() < deadline - 5:
        rng = random.Random(99999)
        ga_adj, ga_cnt = genetic_algorithm(pop_size=30, generations=50, rng=rng)
        if ga_cnt > best_edges and is_k33_free(ga_adj):
            best_edges = ga_cnt
            best_adj = list(ga_adj)
    
    # Final: try to add more edges to the best
    if best_adj is not None:
        best_adj = greedy_add(best_adj)
        best_edges = edge_count(best_adj)
    
    if best_adj is None or not is_k33_free(best_adj):
        # Fallback: construct a valid K3,3-free graph
        # Complete bipartite K10,10 is K3,3-free (actually it contains K3,3!)
        # Wait, K10,10 does contain K3,3. Need a different fallback.
        # Use a cycle graph C20 - definitely K3,3-free with 20 edges
        best_adj = [0] * n
        for i in range(n):
            best_adj[i] |= (1 << ((i + 1) % n))
            best_adj[i] |= (1 << ((i - 1) % n))
        best_edges = n

    edges = mask_to_edges(best_adj)
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")

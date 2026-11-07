"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy:
    1. Use a power-of-2 based set of values to create structured sums/products.
    2. Specifically, use values that are powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.
       Wait, 8192 <= 10000, so 14 distinct powers of 2 work.
       Sums: 2^a + 2^b. Products: 2^a * 2^b = 2^(a+b).
       Products are determined by a+b. Sums are determined by the pair (a,b).
       Actually, let's think differently.
       
    Alternative Strategy:
    Use a small set of values and replicate the structure? No, values must be distinct.
    
    Let's try a greedy/randomized local search approach.
    1. Initialize with random distinct values and random 35 edges.
    2. Compute the score: max(distinct_sums, distinct_products).
    3. Improve by:
       a. Swapping values between vertices.
       b. Swapping edges (replace one edge with another not in the set).
       c. Changing a single value to a new random value.
    
    However, the directive suggests a constraint satisfaction search with Union-Find.
    Let's implement a simpler but effective local search first, as it's robust.
    
    Actually, let's look at the structure.
    If we pick values v_0, ..., v_13.
    Edges (i,j) contribute sum s_ij = v_i + v_j and product p_ij = v_i * v_j.
    We want to minimize K = max(|{s_ij}|, |{p_ij}|).
    
    A good heuristic is to make many sums and products collide.
    Collisions in sums: v_i + v_j = v_k + v_l.
    Collisions in products: v_i * v_j = v_k * v_l.
    
    Let's use a set of values that are arithmetic progressions?
    v_i = a + i*d.
    Sums: 2a + (i+j)d. Distinct sums depend on distinct i+j.
    Products: (a+id)(a+jd) = a^2 + a(i+j)d + ij d^2.
    
    Let's try a specific construction:
    Values: 1, 2, 3, ..., 14.
    Edges: complete graph K_14 has 91 edges. We need 35.
    If we pick edges carefully, can we reduce distinct sums/products?
    
    Let's try a randomized local search.
    """
    
    n = 14
    m = 35
    max_val = 10000
    
    def score(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
    
    # Initial values: random distinct integers
    values = random.sample(range(1, max_val + 1), n)
    
    # Initial edges: random 35 edges
    all_edges = list(combinations(range(n), 2))
    edges = random.sample(all_edges, m)
    
    current_score = score(values, edges)
    
    # Local search
    best_score = current_score
    best_values = values[:]
    best_edges = edges[:]
    
    for iteration in range(5000):
        move_type = random.choice(['value', 'edge', 'swap_val', 'swap_edge'])
        
        new_values = values[:]
        new_edges = edges[:]
        
        if move_type == 'value':
            # Change one random value
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, max_val)
            # Ensure distinct
            if new_val not in new_values or new_values[idx] == new_val:
                if new_val in new_values and new_values[idx] != new_val:
                    continue
                new_values[idx] = new_val
            else:
                continue
        
        elif move_type == 'swap_val':
            # Swap two values
            i, j = random.sample(range(n), 2)
            new_values[i], new_values[j] = new_values[j], new_values[i]
        
        elif move_type == 'edge':
            # Add a random edge not in set, remove a random edge in set
            existing_set = set(tuple(sorted(e)) for e in new_edges)
            available = [e for e in all_edges if tuple(sorted(e)) not in existing_set]
            if not available:
                continue
            new_e = random.choice(available)
            rem_e = random.choice(new_edges)
            new_edges.remove(rem_e)
            new_edges.append(list(new_e))
        
        elif move_type == 'swap_edge':
            # Swap two edges: remove two, add two different
            if len(new_edges) < 2:
                continue
            e1, e2 = random.sample(new_edges, 2)
            existing_set = set(tuple(sorted(e)) for e in new_edges)
            available = [e for e in all_edges if tuple(sorted(e)) not in existing_set]
            if len(available) < 2:
                continue
            new_e1, new_e2 = random.sample(available, 2)
            new_edges.remove(e1)
            new_edges.remove(e2)
            new_edges.append(list(new_e1))
            new_edges.append(list(new_e2))
        
        new_score = score(new_values, new_edges)
        
        if new_score < current_score:
            values = new_values
            edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_values = values[:]
                best_edges = edges[:]
    
    # Return best found
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END

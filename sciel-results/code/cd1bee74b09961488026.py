"""
Optimized search for minimum 5-uniform hypergraphs without Property B.
Uses a fast bitmask-based backtracking checker and a biased Genetic Algorithm.
"""
import time
import random
import itertools
import sys

sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START

def _fast_check_non_2_colorable(n, edges):
    """
    Checks if the hypergraph is NOT 2-colorable.
    Returns True if no 2-coloring exists.
    """
    if not edges:
        return False

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Remove duplicates
    edge_masks = list(set(edge_masks))

    full_mask = (1 << n) - 1

    # Precompute vertex degrees
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    def propagate(color1, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                if (mask & unassigned) == 0:
                    if (mask & color1) == 0:
                        return None, None, True
                    if (mask & ~color1 & full_mask) == 0:
                        return None, None, True
                    continue

                unassigned_in_edge = mask & unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        unassigned ^= unassigned_in_edge
                        changed = True
                    elif (other_4 & color1) == 0:
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True

        return color1, unassigned, False

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False

        if unassigned == 0:
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0:
                    return False
                if (mask & color1) == 0:
                    return False
            return True

        best_v_bit = 0
        best_deg = -1
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if vertex_degree[idx] > best_deg:
                best_deg = vertex_degree[idx]
                best_v_bit = lsb
            m ^= lsb

        v_bit = best_v_bit

        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True
        if backtrack(color1, unassigned ^ v_bit):
            return True

        return False

    initial_color1 = 0
    initial_unassigned = full_mask ^ 1

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True

    if initial_unassigned == 0:
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False

    if backtrack(initial_color1, initial_unassigned):
        return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    rng = random.Random(42)
    pop_size = 60
    
    # Initialize population
    population = []
    
    # Seed with K_8^5 (56 edges)
    n8 = 8
    edges8 = [list(e) for e in itertools.combinations(range(n8), 5)]
    population.append({'n': n8, 'edges': edges8, 'is_non_2c': None, 'fitness': None})
    
    # Seed with random hypergraphs
    for i in range(pop_size - 1):
        n = rng.randint(8, 18)
        num_edges = rng.randint(30, 60)
        all_combos = list(itertools.combinations(range(n), 5))
        if num_edges <= len(all_combos):
            edges = [list(e) for e in rng.sample(all_combos, num_edges)]
        else:
            edges = [list(e) for e in all_combos]
        population.append({'n': n, 'edges': edges, 'is_non_2c': None, 'fitness': None})
        
    best_solution = None
    best_edges_count = float('inf')
    
    def evaluate(ind):
        if ind['is_non_2c'] is None:
            ind['is_non_2c'] = _fast_check_non_2_colorable(ind['n'], ind['edges'])
            ind['fitness'] = (0 if ind['is_non_2c'] else 1, len(ind['edges']))
            
            if ind['is_non_2c'] and len(ind['edges']) < best_edges_count:
                nonlocal best_edges_count
                best_edges_count = len(ind['edges'])
                best_solution = ind
        return ind

    # Initial evaluation
    for ind in population:
        if time.time() > deadline - 10:
            break
        evaluate(ind)
        
    # If no valid solution found yet, force one
    if best_solution is None:
        # Use K_8^5
        ind = population[0]
        evaluate(ind)
        if ind['is_non_2c']:
            best_solution = ind
            best_edges_count = len(ind['edges'])
        else:
            # This shouldn't happen for K_8^5
            pass

    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Sort population
        # We need to ensure all have fitness
        for ind in population:
            if ind['fitness'] is None:
                evaluate(ind)
                
        population.sort(key=lambda x: (x['is_non_2c'] is False, x['fitness'][1]))
        
        # Generate offspring
        offspring = []
        for _ in range(pop_size):
            if time.time() > deadline - 2:
                break
            
            # Select parents
            p1 = population[rng.randint(0, min(15, len(population)-1))]
            p2 = population[rng.randint(0, min(15, len(population)-1))]
            
            # Mutation logic
            n = p1['n']
            edges = [e[:] for e in p1['edges']]
            
            # Bias mutation based on p1's status
            if p1['is_non_2c']:
                # Try to reduce edges
                if edges and rng.random() < 0.7:
                    # Remove edge
                    idx = rng.randint(0, len(edges)-1)
                    edges.pop(idx)
                else:
                    # Add or swap
                    if rng.random() < 0.5 and n >= 5:
                        new_edge = list(rng.sample(range(n), 5))
                        if tuple(new_edge) not in set(map(tuple, edges)):
                            edges.append(new_edge)
                    elif edges:
                        idx = rng.randint(0, len(edges)-1)
                        v_idx = rng.randint(0, 4)
                        new_v = rng.randint(0, n-1)
                        if new_v not in edges[idx]:
                            edges[idx][v_idx] = new_v
            else:
                # Try to add edges to break colorability
                if n >= 5:
                    new_edge = list(rng.sample(range(n), 5))
                    if tuple(new_edge) not in set(map(tuple, edges)):
                        edges.append(new_edge)
                # Also try swapping
                if edges:
                    idx = rng.randint(0, len(edges)-1)
                    v_idx = rng.randint(0, 4)
                    new_v = rng.randint(0, n-1)
                    if new_v not in edges[idx]:
                        edges[idx][v_idx] = new_v
            
            offspring.append({'n': n, 'edges': edges, 'is_non_2c': None, 'fitness': None})
            
            # Evaluate offspring
            if time.time() > deadline - 2:
                break
            evaluate(offspring[-1])
        
        # Combine and select
        combined = population + offspring
        # Sort combined
        # Ensure all evaluated
        for ind in combined:
            if ind['fitness'] is None:
                evaluate(ind)
        combined.sort(key=lambda x: (x['is_non_2c'] is False, x['fitness'][1]))
        population = combined[:pop_size]
        
        # Early stop if we hit 51?
        if best_edges_count <= 51:
            # Keep going for a bit to see if we can get 50
            pass

    if best_solution is None:
        n = 8
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_solution['n'], 'edges': best_solution['edges']}

# EVOLVE-BLOCK-END

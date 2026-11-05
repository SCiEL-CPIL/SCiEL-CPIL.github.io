import random
import time
import math

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute face conditions for each edge
    # For each edge e, store list of 3-tuples of edges that complete a C4 with e
    edge_to_cycle_conditions = {}
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        conditions = []
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            
            e_opp = (min(u_prime, v_prime), max(u_prime, v_prime))
            e_side1 = (min(u, u_prime), max(u, u_prime))
            e_side2 = (min(v, v_prime), max(v, v_prime))
            
            conditions.append((e_opp, e_side1, e_side2))
        edge_to_cycle_conditions[(u, v)] = conditions

    # Helper to check if adding e to a set `chosen` is valid
    # Returns True if no C4 is formed
    def is_valid_addition(e, chosen):
        for (e_opp, e_side1, e_side2) in edge_to_cycle_conditions[e]:
            if e_opp in chosen and e_side1 in chosen and e_side2 in chosen:
                return False
        return True

    # Helper to check if swapping e_out with e_in in `chosen` is valid
    # Resulting set: chosen - {e_out} + {e_in}
    def is_valid_swap(e_in, e_out, chosen):
        if e_in in chosen:
            return False
        # Check if e_in creates a C4 in the new set
        for (e_opp, e_side1, e_side2) in edge_to_cycle_conditions[e_in]:
            # The C4 is formed if all 3 other edges are in the new set
            # New set contains e_in, and (chosen - {e_out})
            # So the 3 other edges must be in (chosen - {e_out})
            # i.e., they are in chosen AND none of them is e_out
            if e_opp in chosen and e_side1 in chosen and e_side2 in chosen:
                # Check if e_out is one of them
                if e_out != e_opp and e_out != e_side1 and e_out != e_side2:
                    return False
        return True

    # Initial solution using Greedy
    random.shuffle(all_edges)
    chosen = set()
    for e in all_edges:
        if is_valid_addition(e, chosen):
            chosen.add(e)
    
    best_chosen = set(chosen)
    best_count = len(chosen)
    
    # Local Search with Simulated Annealing-like acceptance
    # We will try to improve the count.
    # Moves: Add, Swap, Remove
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Try to Add
        # Pick a random edge not in chosen
        # To be efficient, we can just pick random from all_edges and check
        # But as chosen gets large, most edges will be invalid.
        # We can maintain a list of "candidate" edges? 
        # For simplicity, random sampling.
        
        move_type = random.random()
        if move_type < 0.5:
            # Try Add
            e = random.choice(all_edges)
            if e not in chosen and is_valid_addition(e, chosen):
                chosen.add(e)
                if len(chosen) > best_count:
                    best_count = len(chosen)
                    best_chosen = set(chosen)
        elif move_type < 0.9:
            # Try Swap
            e_in = random.choice(all_edges)
            if e_in in chosen:
                continue
            e_out = random.choice(list(chosen))
            if is_valid_swap(e_in, e_out, chosen):
                # Swap is always neutral in size, but might enable future adds
                # Accept swap with probability 1 (or based on T if we had energy)
                # Since size doesn't change, we just do it to explore
                chosen.remove(e_out)
                chosen.add(e_in)
        else:
            # Try Remove (to escape local optima)
            e_out = random.choice(list(chosen))
            chosen.remove(e_out)
            # After removal, we might be able to add something else
            # The next iteration will likely try to Add
            
        # Cool down
        T *= cooling
        
        # If we are stuck (no adds for a while), we might want to do more swaps/removes
        # The random choice above handles this.

    # Final greedy pass on the best found set? 
    # Or just return the best found.
    # Let's do a final greedy pass to see if we can add any more edges to best_chosen
    # This is safe and fast.
    final_chosen = set(best_chosen)
    for e in all_edges:
        if e not in final_chosen and is_valid_addition(e, final_chosen):
            final_chosen.add(e)
    
    if len(final_chosen) > best_count:
        best_chosen = final_chosen
        best_count = len(final_chosen)

    return [[u, v] for u, v in best_chosen]

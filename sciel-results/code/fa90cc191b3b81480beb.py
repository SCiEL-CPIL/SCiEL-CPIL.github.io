"""Optimized program for finding max subset of [1,128] avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct numbers have all pairwise LCMs equal."""
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if time.time() > deadline - 20:
            break
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements
    
    # Build conflict structures using bitmasks for fast checking
    # For each node, store the set of (other1, other2) pairs that form a bad triple
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
    
    # Count conflicts per node
    conflict_count = [len(pairs_by_node[i]) for i in range(n)]
    
    # Precompute bitmask for selected set
    # We'll use a list of booleans for selected status
    
    def can_add(selected_set, node):
        """Check if adding node to selected doesn't complete any bad triple."""
        for other1, other2 in pairs_by_node[node]:
            if other1 in selected_set and other2 in selected_set:
                return False
        return True
    
    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected = set()
        for node in order:
            if time.time() > deadline - 10:
                break
            if can_add(selected, node):
                selected.add(node)
        return selected
    
    def local_search(selected, max_iter=5000):
        """Local search: try to add unselected nodes, then try swaps."""
        selected = set(selected)
        unselected = set(range(n)) - selected
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 15:
                break
            
            # Phase 1: Try to add any unselected node
            for node in list(unselected):
                if time.time() > deadline - 15:
                    break
                if can_add(selected, node):
                    selected.add(node)
                    unselected.remove(node)
                    improved = True
            
            # Phase 2: Try swaps - remove one, add two
            if not improved:
                # Try: for each selected node s, try removing it and adding two unselected
                # This is expensive, so limit attempts
                swap_found = False
                for s in list(selected):
                    if time.time() > deadline - 15:
                        break
                    if swap_found:
                        break
                    # Try removing s
                    selected.remove(s)
                    # Try adding pairs from unselected
                    unselected_list = list(unselected)
                    for i in range(len(unselected_list)):
                        if time.time() > deadline - 15:
                            break
                        u1 = unselected_list[i]
                        if can_add(selected, u1):
                            selected.add(u1)
                            unselected.remove(u1)
                            for j in range(i + 1, len(unselected_list)):
                                if time.time() > deadline - 15:
                                    break
                                u2 = unselected_list[j]
                                if can_add(selected, u2):
                                    selected.add(u2)
                                    unselected.remove(u2)
                                    swap_found = True
                                    improved = True
                                    break
                        if swap_found:
                            break
                        # Undo adding u1
                        selected.discard(u1)
                        unselected.add(u1)
                    # Re-add s if we didn't find a swap
                    if not swap_found:
                        selected.add(s)
                        unselected.discard(s)
        
        return selected
    
    def simulated_annealing(initial_solution, max_iter=10000):
        """Simulated annealing to escape local minima."""
        selected = set(initial_solution)
        unselected = set(range(n)) - selected
        current_size = len(selected)
        best_size = current_size
        best_solution = set(selected)
        
        # Temperature schedule
        t_start = 1.0
        t_end = 0.01
        cooling_rate = (t_end / t_start) ** (1.0 / max_iter)
        temperature = t_start
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            temperature *= cooling_rate
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random unselected node
                if not unselected:
                    continue
                node = random.choice(list(unselected))
                if can_add(selected, node):
                    selected.add(node)
                    unselected.remove(node)
                    new_size = len(selected)
                    if new_size > best_size:
                        best_size = new_size
                        best_solution = set(selected)
                else:
                    # Accept with probability based on temperature
                    delta = -1  # size decreases by 1 if we accept a bad move
                    if random.random() < pow(2.718, delta / temperature):
                        # Accept the bad move (remove a random selected node instead)
                        if selected:
                            remove_node = random.choice(list(selected))
                            selected.remove(remove_node)
                            unselected.add(remove_node)
            else:
                # Try to remove a random selected node and add a different one
                if not selected or not unselected:
                    continue
                remove_node = random.choice(list(selected))
                add_node = random.choice(list(unselected))
                
                selected.remove(remove_node)
                unselected.add(remove_node)
                
                if can_add(selected, add_node):
                    selected.add(add_node)
                    unselected.remove(add_node)
                    new_size = len(selected)
                    if new_size > best_size:
                        best_size = new_size
                        best_solution = set(selected)
                else:
                    # Undo
                    selected.add(remove_node)
                    unselected.remove(remove_node)
        
        return best_solution
    
    # Multiple restarts with different orderings
    best_solution = set()
    best_size = 0
    
    # Ordering 1: by conflict count (ascending)
    order1 = sorted(range(n), key=lambda i: conflict_count[i])
    sol1 = greedy_init(order1)
    sol1 = local_search(sol1, max_iter=3000)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_solution = sol1
    
    # Ordering 2: reverse
    order2 = list(reversed(order1))
    sol2 = greedy_init(order2)
    sol2 = local_search(sol2, max_iter=3000)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_solution = sol2
    
    # Ordering 3: natural order
    order3 = list(range(n))
    sol3 = greedy_init(order3)
    sol3 = local_search(sol3, max_iter=3000)
    if len(sol3) > best_size:
        best_size = len(sol3)
        best_solution = sol3
    
    # Ordering 4: reverse natural
    order4 = list(range(n - 1, -1, -1))
    sol4 = greedy_init(order4)
    sol4 = local_search(sol4, max_iter=3000)
    if len(sol4) > best_size:
        best_size = len(sol4)
        best_solution = sol4
    
    # Randomized greedy with local search (inspiration from random restarts)
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        sol_r = greedy_init(random_order)
        sol_r = local_search(sol_r, max_iter=2000)
        if len(sol_r) > best_size:
            best_size = len(sol_r)
            best_solution = sol_r
    
    # Simulated annealing on best solution (inspiration from bitmask-based SA)
    if time.time() < deadline - 25:
        sa_solution = simulated_annealing(best_solution, max_iter=5000)
        if len(sa_solution) > best_size:
            best_size = len(sa_solution)
            best_solution = sa_solution
    
    # Additional local search passes on best solution
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            improved_sol = local_search(best_solution, max_iter=2000)
            if len(improved_sol) > len(best_solution):
                best_solution = improved_sol
    
    # Final attempt: try to add any remaining nodes
    if time.time() < deadline - 5:
        for node in range(n):
            if node not in best_solution and can_add(best_solution, node):
                best_solution.add(node)
    
    return [i + 1 for i in sorted(best_solution)]

# EVOLVE-BLOCK-END

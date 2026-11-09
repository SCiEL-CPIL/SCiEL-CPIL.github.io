"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching the solution order
_solution_cache = {}
_cache_key = None


def _beam_search(weights, values, capacity, beam_width=50, max_steps=50):
    """Beam search for 0/1 knapsack. Returns list of indices in order of selection."""
    n = len(weights)
    if n == 0:
        return []
    
    # Each beam state: (total_value, total_weight, chosen_indices_set)
    # We'll keep beam as list of (value, weight, chosen_set)
    beam = [(0.0, 0.0, frozenset())]
    
    for step in range(max_steps):
        new_beam = []
        for val, wt, chosen in beam:
            # Try adding each item not yet chosen
            for i in range(n):
                if i in chosen:
                    continue
                new_wt = wt + weights[i]
                if new_wt <= capacity:
                    new_val = val + values[i]
                    new_chosen = chosen | {i}
                    new_beam.append((new_val, new_wt, new_chosen))
        
        if not new_beam:
            break
        
        # Keep top beam_width by value
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        # If no more items can be added to the best state, we might stop
        # But we need to continue until no more improvements
    
    # Get the best state
    if not beam:
        return []
    best_val, best_wt, best_chosen = beam[0]
    
    # Now we need to determine the order. We can do a greedy reconstruction.
    # Actually, for the iterative interface, we need an order.
    # Let's do: greedily pick from best_chosen in order of density, but that might not match.
    # Better: reconstruct the order by doing a DFS or just pick in order of addition.
    # For simplicity, return the items in best_chosen sorted by value/weight descending.
    # But the order matters for the iterative process.
    
    # Actually, the order doesn't matter for the final set, but the iterative process
    # requires that each picked item fits at the time of picking.
    # So we need an order where each prefix fits.
    # A safe order: sort by weight ascending, then pick.
    # Or: do a greedy pick from best_chosen.
    
    chosen_list = list(best_chosen)
    # Sort by value/weight descending for a reasonable order
    chosen_list.sort(key=lambda i: values[i] / weights[i] if weights[i] > 0 else float('inf'), reverse=True)
    
    return chosen_list


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _solution_cache, _cache_key
    
    n = len(remaining_weights)
    
    # If this is the first call (full problem), solve it
    if _cache_key is None or _cache_key != n:
        # First call: solve the full problem
        weights = np.array(remaining_weights, dtype=np.float64)
        values = np.array(remaining_values, dtype=np.float64)
        capacity = float(remaining_capacity)
        
        # Run beam search
        solution = _beam_search(weights, values, capacity, beam_width=100, max_steps=100)
        
        # Store the solution
        _solution_cache = solution
        _cache_key = n
        _solution_cache_index = 0
        return solution[0] if solution else -1
    
    # Subsequent calls: return next item from cached solution that is still available
    # We need to track which items have been picked
    if not hasattr(select_next_item, '_picked_set'):
        select_next_item._picked_set = set()
        select_next_item._solution = _solution_cache
        select_next_item._idx = 0
    
    # Find the next item in the solution that is in the remaining set
    remaining_set = set(range(n))
    for i in select_next_item._solution:
        if i in remaining_set and i not in select_next_item._picked_set:
            select_next_item._picked_set.add(i)
            return i
    
    return -1
# EVOLVE-BLOCK-END

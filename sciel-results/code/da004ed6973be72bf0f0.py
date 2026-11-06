import time
import sys
import random
from math import gcd
from functools import reduce

# Set recursion limit
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    
    Strategy: Multi-strategy search with greedy construction, local search improvement,
    and randomized restarts to escape local optima.
    """
    
    # Precompute LCM of 1..22
    L = reduce(_lcm, range(1, 23), 1)
    
    # Weights w[i] = L // i
    weights = {i: L // i for i in range(1, 23)}
    
    def _get_sums(indices):
        """Compute all subset sums for a list of indices."""
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            if new_sums & sums:
                return None  # Collision
            sums = sums.union(new_sums)
        return sums
    
    def _is_valid(indices):
        """Check if indices form a dissociated set."""
        return _get_sums(indices) is not None
    
    def _greedy_build(order):
        """Build a set using greedy approach with given order of indices."""
        selected = []
        sums = {0}
        for idx in order:
            w = weights[idx]
            # Check collision
            collision = False
            for s in sums:
                if (s + w) in sums:
                    collision = True
                    break
            if not collision:
                selected.append(idx)
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w)
                sums = new_sums
        return selected, sums
    
    def _local_search_improve(selected, sums, max_iters=1000):
        """Try to improve the set by swapping elements."""
        if not selected:
            return selected, sums
        
        all_indices = set(range(1, 23))
        current_selected = set(selected)
        best_selected = list(selected)
        best_sums = sums
        best_size = len(selected)
        
        for _ in range(max_iters):
            if _time_left() < 1.0:
                break
            
            improved = False
            # Try removing one element and adding another
            for remove_idx in list(current_selected):
                # Try adding each non-selected index
                for add_idx in all_indices - current_selected:
                    new_selected = current_selected - {remove_idx} | {add_idx}
                    new_selected_list = list(new_selected)
                    new_sums = _get_sums(new_selected_list)
                    if new_sums is not None and len(new_selected_list) > best_size:
                        best_selected = new_selected_list
                        best_sums = new_sums
                        best_size = len(new_selected_list)
                        current_selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding an element if possible
                for add_idx in all_indices - current_selected:
                    new_selected = current_selected | {add_idx}
                    new_selected_list = list(new_selected)
                    new_sums = _get_sums(new_selected_list)
                    if new_sums is not None:
                        best_selected = new_selected_list
                        best_sums = new_sums
                        best_size = len(new_selected_list)
                        current_selected = new_selected
                        improved = True
                        break
                
                if not improved:
                    break
        
        return best_selected, best_sums
    
    def _backtrack_find(k, order, max_nodes=50000):
        """Backtracking search for a set of size k using given order."""
        nodes_visited = [0]
        
        def backtrack(start_pos, current_set, current_sums):
            if _time_left() < 0.5 or nodes_visited[0] > max_nodes:
                return None
            
            nodes_visited[0] += 1
            
            if len(current_set) == k:
                return list(current_set)
            
            remaining_needed = k - len(current_set)
            available = len(order) - start_pos
            if available < remaining_needed:
                return None
            
            for i in range(start_pos, len(order)):
                idx = order[i]
                w = weights[idx]
                
                # Check collision
                collision = False
                for s in current_sums:
                    if (s + w) in current_sums:
                        collision = True
                        break
                
                if collision:
                    continue
                
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + w)
                
                res = backtrack(i + 1, current_set + [idx], new_sums)
                if res is not None:
                    return res
                    
            return None
        
        return backtrack(0, [], {0})
    
    # Strategy 1: Multiple greedy orderings
    best_result = None
    best_size = 0
    
    # Different orderings to try
    orderings = []
    
    # Ascending index
    orderings.append(list(range(1, 23)))
    
    # Descending index
    orderings.append(list(range(22, 0, -1)))
    
    # By weight descending (smallest indices first)
    orderings.append(sorted(range(1, 23), key=lambda x: -weights[x]))
    
    # By weight ascending
    orderings.append(sorted(range(1, 23), key=lambda x: weights[x]))
    
    # Random orderings
    for _ in range(20):
        if _time_left() < 5.0:
            break
        rand_order = list(range(1, 23))
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for order in orderings:
        if _time_left() < 2.0:
            break
        
        selected, sums = _greedy_build(order)
        if sums is not None:
            # Try local search improvement
            improved_selected, improved_sums = _local_search_improve(selected, sums, max_iters=200)
            if improved_sums is not None and len(improved_selected) > best_size:
                best_size = len(improved_selected)
                best_result = improved_selected
    
    # Strategy 2: Backtracking for larger sets
    if best_size < 12 and _time_left() > 10.0:
        for k in range(min(14, best_size + 3), best_size, -1):
            if _time_left() < 5.0:
                break
            
            # Try backtracking with different orderings
            for order in orderings[:5]:  # Try first 5 orderings
                if _time_left() < 2.0:
                    break
                
                res = _backtrack_find(k, order, max_nodes=20000)
                if res is not None and _is_valid(res):
                    if len(res) > best_size:
                        best_size = len(res)
                        best_result = res
                    break
    
    # Strategy 3: Randomized local search from scratch
    if _time_left() > 5.0:
        for restart in range(50):
            if _time_left() < 2.0:
                break
            
            # Random starting set
            rand_set = list(range(1, 23))
            random.shuffle(rand_set)
            start_set = rand_set[:random.randint(3, 8)]
            
            # Build greedily from this start
            selected = []
            sums = {0}
            for idx in start_set:
                w = weights[idx]
                collision = False
                for s in sums:
                    if (s + w) in sums:
                        collision = True
                        break
                if not collision:
                    selected.append(idx)
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + w)
                    sums = new_sums
            
            if sums is not None:
                # Local search
                improved_selected, improved_sums = _local_search_improve(selected, sums, max_iters=500)
                if improved_sums is not None and len(improved_selected) > best_size:
                    best_size = len(improved_selected)
                    best_result = improved_selected
    
    # Return best result found
    if best_result is not None:
        return best_result
    
    # Fallback: simple greedy
    selected, sums = _greedy_build(list(range(1, 23)))
    if selected:
        return selected
    
    return [1]

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    L = reduce(_lcm, range(1, 23), 1)
    sums = {0}
    valid = True
    for x in s:
        w = L // x
        new_sums = set()
        for ss in sums:
            ns = ss + w
            new_sums.add(ns)
        if new_sums & sums:
            valid = False
            break
        sums = sums.union(new_sums)
    print(f"Valid: {valid}")
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import defaultdict

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precompute range
    range_1_N = list(range(1, N + 1))

    def get_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def can_add(x, selected, sums):
        """Check if x can be added to selected without violating Sidon property."""
        if 2 * x in sums:
            return False
        for a in selected:
            if x + a in sums:
                return False
        return True

    def is_maximal(selected, sums):
        """Check if no element from [1, N] can be added."""
        sel_set = set(selected)
        for x in range_1_N:
            if x in sel_set:
                continue
            if can_add(x, selected, sums):
                return False
        return True

    def greedy_construct(order):
        """Construct a maximal Sidon set using a given candidate order."""
        selected = []
        sums = set()
        sel_set = set()
        
        for x in order:
            if x in sel_set:
                continue
            if can_add(x, selected, sums):
                selected.append(x)
                sel_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        return selected, sums

    def local_search_minimize(selected, sums, time_limit):
        """Try to remove elements while maintaining maximality."""
        selected = list(selected)
        sums = get_sums(selected)
        sel_set = set(selected)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                x = selected[i]
                # Tentatively remove x
                new_selected = selected[:i] + selected[i+1:]
                if not new_selected:
                    continue
                
                new_sums = get_sums(new_selected)
                
                # Check if maximal
                if is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    sel_set = set(selected)
                    changed = True
                    break
        
        return selected

    # 1. Generate an initial valid maximal Sidon set using greedy
    # Use descending order to get a reasonably small set quickly
    initial_order = list(range(N, 0, -1))
    best_selected, best_sums = greedy_construct(initial_order)
    
    # Ensure it is maximal (greedy with full range usually is, but verify)
    if not is_maximal(best_selected, best_sums):
        # Extend if needed
        sel_set = set(best_selected)
        for x in range_1_N:
            if x not in sel_set and can_add(x, best_selected, best_sums):
                best_selected.append(x)
                sel_set.add(x)
                best_sums.add(2 * x)
                for a in best_selected[:-1]:
                    best_sums.add(x + a)
        best_sums = get_sums(best_selected)

    best_size = len(best_selected)
    
    # 2. Simulated Annealing to minimize size
    # State: (selected_list, sums_set)
    # Move: Swap an element 'out' with an element 'in'
    #   - Must maintain Sidon property
    #   - Maximality is checked at the end or periodically
    
    current_selected = list(best_selected)
    current_sums = get_sums(current_selected)
    current_sel_set = set(current_selected)
    
    best_selected_sa = list(current_selected)
    best_size_sa = len(current_selected)
    
    # Precompute list of all elements for sampling
    all_elements = range_1_N
    
    # Annealing parameters
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    # We will perform a fixed number of steps or until time limit
    steps = 0
    max_steps = 200000  # Cap steps to avoid infinite loop if time check is slow
    
    while time.time() < deadline and T > T_min:
        steps += 1
        
        # 1. Pick an element to remove (current)
        if not current_selected:
            break
        idx_out = random.randint(0, len(current_selected) - 1)
        x_out = current_selected[idx_out]
        
        # 2. Pick an element to add (candidate)
        # It must not be in current_selected
        # We pick randomly from [1, N]
        x_in = random.choice(all_elements)
        if x_in in current_sel_set:
            continue
            
        # 3. Check Sidon validity of the swap
        # Remove x_out, add x_in
        # New set = (current_selected - {x_out}) U {x_in}
        
        # Efficient check:
        # Sums affected by removing x_out:
        #   2*x_out
        #   x_out + y for all y in current_selected, y != x_out
        # Sums affected by adding x_in:
        #   2*x_in
        #   x_in + y for all y in (current_selected - {x_out})
        
        # Let's just compute the new sums set for simplicity if size is small enough,
        # or use a delta approach. Given N=256, max size ~16-20, O(n^2) is fast.
        
        new_selected = [x for x in current_selected if x != x_out]
        new_selected.append(x_in)
        
        # Check Sidon property
        valid = True
        sums_new = set()
        for i, a in enumerate(new_selected):
            for b in new_selected[i:]:
                s = a + b
                if s in sums_new:
                    valid = False
                    break
                sums_new.add(s)
            if not valid:
                break
        
        if not valid:
            # Reject move
            # In SA, we still cool down
            T *= cooling
            continue
        
        # Move is valid Sidon. 
        # Now, do we accept it?
        # Objective: Minimize size.
        # But we also want to maintain maximality.
        # If we accept non-maximal sets, we might get stuck or end with invalid result.
        # Strategy: 
        #   - If new set is maximal: Accept (improvement or neutral in size, but keeps validity)
        #   - If new set is NOT maximal: 
        #       - Only accept if it reduces size? No, that risks losing maximality permanently.
        #       - Better: Only accept moves that keep the set maximal?
        #         This is equivalent to local search with swaps.
        #       - Alternative: Accept non-maximal moves with low probability (exploration), 
        #         but always try to restore maximality?
        
        # Let's stick to moves that preserve maximality first (Local Search with Swaps).
        # This is safer. If we get stuck, we can introduce probabilistic acceptance of 
        # non-maximal states later, but for now, let's see if swaps alone help.
        
        if is_maximal(new_selected, sums_new):
            # Accept the move
            current_selected = new_selected
            current_sums = sums_new
            current_sel_set = set(current_selected)
            
            if len(current_selected) < best_size_sa:
                best_size_sa = len(current_selected)
                best_selected_sa = list(current_selected)
                # Early exit if very small
                if best_size_sa <= 15:
                    break
        else:
            # Reject non-maximal moves for now (greedy swap local search)
            pass
            
        T *= cooling

    # 3. Final Local Search (Removals) on the best found set
    # The SA might have found a set that allows removals now
    final_selected = best_selected_sa
    final_sums = get_sums(final_selected)
    
    # Try to remove elements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(final_selected)):
            x = final_selected[i]
            new_selected = final_selected[:i] + final_selected[i+1:]
            if not new_selected:
                continue
            new_sums = get_sums(new_selected)
            if is_maximal(new_selected, new_sums):
                final_selected = new_selected
                final_sums = new_sums
                changed = True
                break

    # Verify final result
    if is_valid_sidon(final_selected) and is_maximal(final_selected, get_sums(final_selected)):
        if len(final_selected) < best_size:
            best_selected = final_selected
            best_size = len(final_selected)
    else:
        # Fallback to original if something went wrong
        pass

    # If we didn't improve with SA, return original best
    if best_size_sa < best_size:
        # Re-verify best_selected_sa
        if is_valid_sidon(best_selected_sa) and is_maximal(best_selected_sa, get_sums(best_selected_sa)):
            best_selected = best_selected_sa
            best_size = best_size_sa

    return best_selected

def is_valid_sidon(selected):
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)

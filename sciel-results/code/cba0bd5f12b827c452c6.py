import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin before 60s hard limit
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    num_moduli = len(moduli)
    
    # Precompute gcd table for fast lookup
    gcd_table = [[0] * (max_modulus + 1) for _ in range(max_modulus + 1)]
    for i in range(2, max_modulus + 1):
        for j in range(2, max_modulus + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def check_valid(selected, r, m):
        """Check if (r, m) is compatible with all in selected."""
        for r_j, m_j in selected:
            g = gcd_table[m][m_j]
            if r % g == r_j % g:
                return False
        return True
    
    def find_valid_residue(selected, m, randomize=False):
        """Find any valid residue for m given selected classes."""
        valid = []
        for r in range(m):
            if check_valid(selected, r, m):
                valid.append(r)
        if not valid:
            return None
        if randomize:
            return random.choice(valid)
        return valid[0]
    
    def greedy(order, randomize=False):
        """Greedy: process moduli in order, add if possible."""
        selected = []
        selected_moduli = set()
        for m in order:
            if time.time() > start_time + max_time:
                break
            if m in selected_moduli:
                continue
            r = find_valid_residue(selected, m, randomize)
            if r is not None:
                selected.append((r, m))
                selected_moduli.add(m)
        return selected
    
    def local_search_fill(selected, time_limit):
        """Try to add missing moduli in random order."""
        best = selected[:]
        selected_moduli = set(m for r, m in best)
        
        missing = [m for m in moduli if m not in selected_moduli]
        random.shuffle(missing)
        
        for m in missing:
            if time.time() > time_limit:
                break
            r = find_valid_residue(best, m, randomize=True)
            if r is not None:
                best.append((r, m))
                selected_moduli.add(m)
        
        return best
    
    def branch_and_bound(selected, remaining, bound, time_limit):
        """
        Branch and bound to try to improve the solution.
        selected: current list of (r, m) pairs
        remaining: list of (m) moduli not yet in selected
        bound: current best size to beat
        Returns improved solution or None if no improvement found within time
        """
        if time.time() > time_limit:
            return None
        
        # If we can't beat the bound even by adding all remaining, prune
        if len(selected) + len(remaining) <= bound:
            return None
        
        if not remaining:
            return selected if len(selected) > bound else None
        
        # Choose next modulus to branch on
        # Heuristic: pick the one with fewest valid residues (most constrained)
        best_idx = -1
        best_count = float('inf')
        
        for i, m in enumerate(remaining):
            count = 0
            for r in range(m):
                if check_valid(selected, r, m):
                    count += 1
                    if count >= best_count:
                        break
            if count < best_count:
                best_count = count
                best_idx = i
                if count == 0:
                    break
        
        if best_idx == -1:
            return None
        
        m = remaining[best_idx]
        remaining_rest = remaining[:best_idx] + remaining[best_idx+1:]
        
        # Try each valid residue for this modulus
        valid_residues = []
        for r in range(m):
            if check_valid(selected, r, m):
                valid_residues.append(r)
        
        # Shuffle for randomization
        random.shuffle(valid_residues)
        
        best_result = None
        for r in valid_residues:
            new_selected = selected + [(r, m)]
            result = branch_and_bound(new_selected, remaining_rest, bound, time_limit)
            if result is not None:
                if best_result is None or len(result) > len(best_result):
                    best_result = result
                # If we found a solution that beats the bound, return it
                # (could continue to find even better, but let's be conservative)
        
        return best_result
    
    def local_search_remove_add(selected, time_limit):
        """
        Try to improve by removing one element and adding two others.
        This is a more powerful move than just filling.
        """
        best = selected[:]
        best_size = len(best)
        selected_moduli = set(m for r, m in best)
        
        # Try each element for removal
        for i in range(len(best)):
            if time.time() > time_limit:
                break
            
            temp = best[:i] + best[i+1:]
            temp_moduli = set(m for r, m in temp)
            
            # Try to add two missing moduli
            missing = [m for m in moduli if m not in temp_moduli]
            random.shuffle(missing)
            
            added = 0
            new_elements = []
            for m in missing:
                if added >= 2:
                    break
                r = find_valid_residue(temp, m, randomize=True)
                if r is not None:
                    # Also check against new_elements
                    compatible = True
                    for r2, m2 in new_elements:
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            compatible = False
                            break
                    if compatible:
                        new_elements.append((r, m))
                        added += 1
            
            if added == 2:
                candidate = temp + new_elements
                if len(candidate) > best_size:
                    best = candidate
                    best_size = len(best)
                    selected_moduli = set(m for r, m in best)
                    # Restart the loop with the new best
                    return best
        
        return best
    
    best_solution = []
    best_size = 0
    
    # Initial greedy with fixed order to get a baseline
    current_set = greedy(moduli, randomize=False)
    best_solution = current_set
    best_size = len(current_set)
    
    # Randomized local search / greedy with restarts
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        random.shuffle(moduli)
        current_set = greedy(moduli, randomize=True)
        
        # Apply local search fill
        current_set = local_search_fill(current_set, start_time + max_time)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = current_set[:]
        
        # Try branch and bound to improve further
        selected_moduli = set(m for r, m in best_solution)
        remaining = [m for m in moduli if m not in selected_moduli]
        
        if remaining and time.time() - start_time < max_time - 5:  # Leave 5s margin
            # Use a tighter time limit for B&B
            bb_limit = min(start_time + max_time - 2, time.time() + 5)
            improved = branch_and_bound(best_solution[:], remaining, best_size, bb_limit)
            if improved is not None:
                best_solution = improved
                best_size = len(best_solution)
        
        # Try local search remove-add
        if time.time() - start_time < max_time - 3:
            la_limit = min(start_time + max_time - 1, time.time() + 3)
            improved = local_search_remove_add(best_solution[:], la_limit)
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = improved
    
    if not best_solution:
        best_solution = [(0, 2)]
    
    return [[r, m] for r, m in best_solution]

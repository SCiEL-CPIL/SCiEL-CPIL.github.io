"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Precompute gcd table for [1, 128]
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c, selected_set):
        """Check if a, b, c form a bad triple (three pairwise gcds equal)."""
        gab = gcd_table[a][b]
        gac = gcd_table[a][c]
        gbc = gcd_table[b][c]
        return gab == gac == gbc

    def is_valid_addition(x, selected):
        """Check if adding x to selected keeps the set valid."""
        for a, b in combinations(selected, 2):
            if is_bad_triple(a, b, x, selected):
                return False
        return True

    def count_bad_triples(selected):
        """Count number of bad triples in selected."""
        count = 0
        for a, b, c in combinations(selected, 3):
            if is_bad_triple(a, c, b, selected):
                count += 1
        return count

    def try_add(x, selected):
        """Try to add x, returning (success, new_selected)."""
        if is_valid_addition(x, selected):
            return True, selected + [x]
        return False, selected

    # Strategy: Greedy initialization + local search with swaps
    # First, greedy add in increasing order
    selected = []
    for x in range(1, n + 1):
        if is_valid_addition(x, selected):
            selected.append(x)
    
    best_selected = list(selected)
    best_size = len(selected)

    # Local search: try to swap out elements that block many additions
    # and replace with elements that were previously blocked
    blocked = [x for x in range(1, n + 1) if x not in selected]
    
    def get_blocking_score(x, selected):
        """How many elements in blocked would be unblocked if we removed x?"""
        unblocked_count = 0
        for y in blocked:
            # Check if y is only blocked by x (and other elements)
            # Simplified: check if removing x allows y to be added
            pass
        return 0

    # More effective local search: try all single swaps
    # For each element in selected, try removing it and adding a better element
    while time.time() < deadline:
        improved = False
        selected_set = set(selected)
        
        # Try removing each element and adding a replacement
        for i, rem in enumerate(selected):
            if time.time() > deadline:
                break
            
            # Try adding each blocked element in place of rem
            for add in blocked:
                if add in selected_set:
                    continue
                
                # Check if replacing rem with add is valid
                new_selected = [x for j, x in enumerate(selected) if j != i]
                new_selected.append(add)
                
                # Validate
                valid = True
                for a, b in combinations(new_selected, 2):
                    # Check with each other element
                    for c in new_selected:
                        if c != a and c != b:
                            if is_bad_triple(a, b, c, new_selected):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid and len(new_selected) > len(selected):
                    # This is an improvement (shouldn't happen with single swap of same size)
                    selected = new_selected
                    selected_set = set(selected)
                    blocked = [x for x in range(1, n + 1) if x not in selected_set]
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            
            if improved:
                if len(selected) > best_size:
                    best_selected = list(selected)
                    best_size = len(selected)
                break
        
        if not improved:
            # Try 2-for-1 or 1-for-2 swaps? Or just try different removals
            # Try removing two elements and adding one that was blocked
            # This is more expensive, so only do it if we have time
            if time.time() < deadline - 10:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(selected)):
                        if time.time() > deadline:
                            break
                        rem1 = selected[i]
                        rem2 = selected[j]
                        
                        for add in blocked:
                            if add in selected_set:
                                continue
                            
                            new_selected = [x for k, x in enumerate(selected) if k != i and k != j]
                            new_selected.append(add)
                            
                            valid = True
                            for a, b in combinations(new_selected, 2):
                                for c in new_selected:
                                    if c != a and c != b:
                                        if is_bad_triple(a, b, c, new_selected):
                                            valid = False
                                            break
                                if not valid:
                                    break
                            
                            if valid and len(new_selected) > len(selected):
                                selected = new_selected
                                selected_set = set(selected)
                                blocked = [x for x in range(1, n + 1) if x not in selected_set]
                                improved = True
                                if len(selected) > best_size:
                                    best_selected = list(selected)
                                    best_size = len(selected)
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if not improved:
            # Stuck, try a different greedy order
            # Randomized greedy restart
            import random
            random.seed(int(time.time() * 1000) % 10000)
            order = list(range(1, n + 1))
            random.shuffle(order)
            
            new_selected = []
            for x in order:
                if is_valid_addition(x, new_selected):
                    new_selected.append(x)
            
            if len(new_selected) > len(selected):
                selected = new_selected
                selected_set = set(selected)
                blocked = [x for x in range(1, n + 1) if x not in selected_set]
                if len(selected) > best_size:
                    best_selected = list(selected)
                    best_size = len(selected)
            else:
                # No improvement after multiple attempts, break
                break

    return best_selected
# EVOLVE-BLOCK-END

import time
import random
from math import gcd
from itertools import combinations

def construct():
    # Strict deadline to avoid timeout (90s limit, use 85s)
    deadline = time.time() + 85
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def is_compatible(x, selected):
        """
        Check if x can be added to selected without creating a bad triple.
        selected is a set.
        Optimization: Group elements in selected by their gcd with x.
        """
        if not selected:
            return True
        
        # Map gcd value -> list of elements in selected having that gcd with x
        groups = {}
        for y in selected:
            g = gcd_table[x][y]
            if g not in groups:
                groups[g] = []
            groups[g].append(y)
        
        # For each group, check if any pair has gcd equal to the group's gcd
        for g, elems in groups.items():
            if len(elems) < 2:
                continue
            # Check all pairs in this group
            # If len is small, O(k^2) is fine.
            # If len is large, it's expensive, but rare for specific x.
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if gcd_table[elems[i]][elems[j]] == g:
                        return False
        return True

    def get_bad_triples_count(selected):
        """Count bad triples in selected (for debugging or annealing, maybe not needed if we use hard constraints)"""
        count = 0
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                for k in range(j + 1, len(sel_list)):
                    if is_bad_triple(sel_list[i], sel_list[j], sel_list[k]):
                        count += 1
        return count

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    max_restarts = 20
    # Adjust restarts based on time? 
    # Each greedy+LS takes ~1-2s. 20 restarts = 20-40s. Safe.
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # 1. Greedy Construction
        # Randomize the order slightly to escape local optima
        # Sort by degree, but add noise
        degrees = [0] * (N + 1)
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                for c in range(b + 1, N + 1):
                    if is_bad_triple(a, b, c):
                        degrees[a] += 1
                        degrees[b] += 1
                        degrees[c] += 1
        
        # Add random noise to degrees to break ties and explore different paths
        noisy_degrees = []
        for x in numbers:
            noise = random.uniform(0, 1)
            noisy_degrees.append((degrees[x] + noise, x))
        
        sorted_numbers = [x for _, x in sorted(noisy_degrees)]
        
        selected = set()
        for x in sorted_numbers:
            if time.time() > deadline - 5:
                break
            if is_compatible(x, selected):
                selected.add(x)
                
        # 2. Local Search
        # Phase A: Add any compatible number
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in numbers:
                if x not in selected:
                    if is_compatible(x, selected):
                        selected.add(x)
                        improved = True
                        break # Restart loop after addition
        
        # Phase B: 1-for-1 Swaps
        # Try to swap out one element to allow adding another? 
        # Or just swap to maintain size but potentially allow future additions?
        # Actually, if no addition is possible, 1-for-1 swap doesn't increase size.
        # It might unblock a future addition if we do it repeatedly? 
        # But 1-for-2 is more direct for size increase.
        
        # Phase C: 1-for-2 Swaps
        # Try to remove 1 element y and add 2 elements x1, x2
        ls_improved = True
        while ls_improved and time.time() < deadline - 5:
            ls_improved = False
            # Sample a few y's to remove
            # If set is large, sampling is better than iterating all
            sel_list = list(selected)
            random.shuffle(sel_list)
            
            for y in sel_list[:10]: # Try top 10 random removals
                if time.time() > deadline - 5:
                    break
                
                temp_selected = selected - {y}
                
                # Find candidates x that are compatible with temp_selected
                candidates = []
                for x in numbers:
                    if x == y or x in selected:
                        continue
                    if is_compatible(x, temp_selected):
                        candidates.append(x)
                
                if len(candidates) < 2:
                    continue
                    
                # Try pairs from candidates
                # Limit pairs to avoid O(N^2) blowup if candidates is large
                # Shuffle candidates to randomize
                random.shuffle(candidates)
                max_pairs_to_check = 50
                checked = 0
                
                for i in range(len(candidates)):
                    if checked >= max_pairs_to_check:
                        break
                    x1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        if checked >= max_pairs_to_check:
                            break
                        x2 = candidates[j]
                        checked += 1
                        
                        # Check if {x1, x2} is compatible with each other and temp_selected
                        # We know x1, x2 are compatible with temp_selected individually.
                        # We need to check if they form a bad triple with any z in temp_selected.
                        # And if x1, x2 form a bad triple with each other? No, need 3 distinct.
                        # So check: for all z in temp_selected, is (x1, x2, z) bad?
                        
                        valid = True
                        for z in temp_selected:
                            if is_bad_triple(x1, x2, z):
                                valid = False
                                break
                            
                        if valid:
                            # Accept swap
                            selected = temp_selected | {x1, x2}
                            ls_improved = True
                            break
                    if ls_improved:
                        break
                if ls_improved:
                    break
            
        # Update best
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
        # Optional: If we have time, do a final polish on best_set?
        # The loop above already does LS.
        
    return list(best_set)

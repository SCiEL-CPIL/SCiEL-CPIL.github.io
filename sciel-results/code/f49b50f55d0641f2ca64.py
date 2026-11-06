from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    all_elements = list(range(1, 25))
    
    # Step 1: Find all minimal forbidden sets
    # A forbidden set C is such that sum(1/y for y in C\{max(C)}) = 1/max(C)
    # Actually, the target x is the one whose reciprocal is summed.
    # So C = {x} U T, sum(1/y for y in T) = 1/x.
    
    forbidden_sets = []
    
    for x in all_elements:
        target = Fraction(1, x)
        candidates = [y for y in all_elements if y != x]
        
        # Find all subsets T of candidates summing to target
        # We use backtracking
        valid_Ts = []
        
        def find_Ts(start, current_sum, current_T):
            if current_sum == target:
                valid_Ts.append(tuple(sorted(current_T)))
                return
            if current_sum > target:
                return
            # Pruning: if current_sum + sum(1/y for y in candidates[start:]) < target
            # We can precompute suffix sums of reciprocals for candidates
            # But candidates changes with x.
            # Let's just do simple backtracking. N is small.
            
            for i in range(start, len(candidates)):
                y = candidates[i]
                find_Ts(i + 1, current_sum + Fraction(1, y), current_T + [y])
                
        find_Ts(0, Fraction(0, 1), [])
        
        # Filter minimal T
        # T1 is minimal if no proper subset T2 of T1 sums to target
        minimal_Ts = []
        for T in valid_Ts:
            is_minimal = True
            # Check all proper subsets
            # Number of subsets is 2^len(T). T is likely small.
            # We can just check if any T' in valid_Ts is a proper subset of T
            for T2 in valid_Ts:
                if T2 != T and set(T2).issubset(set(T)):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_Ts.append(T)
                
        for T in minimal_Ts:
            C = tuple(sorted([x] + list(T)))
            forbidden_sets.append(set(C))
            
    # Remove duplicates
    forbidden_sets = list(set(forbidden_sets))
    
    # Step 2: Find max subset S of [1, 24] such that no C in forbidden_sets is subset of S
    # Use backtracking
    
    def find_max_set():
        best_size = 0
        best_set = []
        
        # Sort forbidden sets by size?
        # We want to avoid them.
        
        def backtrack(start_idx, current_set, current_size):
            nonlocal best_size, best_set
            
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
                
            if start_idx >= 24:
                return
            
            # Pruning: if current_size + (24 - start_idx) <= best_size, return
            if current_size + (24 - start_idx) <= best_size:
                return
                
            for i in range(start_idx, 24):
                x = i + 1
                # Check if adding x creates a forbidden set
                new_set = current_set | {x}
                
                valid = True
                for C in forbidden_sets:
                    if C.issubset(new_set):
                        valid = False
                        break
                        
                if valid:
                    backtrack(i + 1, new_set, current_size + 1)
                    
        backtrack(0, set(), 0)
        return best_set
        
    return find_max_set()
# EVOLVE-BLOCK-END

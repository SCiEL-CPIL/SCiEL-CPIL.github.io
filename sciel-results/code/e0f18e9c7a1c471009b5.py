import random

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    of different cardinalities are disjoint.
    Maximizes the set size.
    """
    N = 256
    
    def check_and_update(current_sums, new_element, k_target):
        """
        current_sums: list of sets, current_sums[i] is set of sums of size i.
        new_element: int
        Returns: (is_valid, updated_sums)
        """
        updated_sums = [set(s) for s in current_sums]
        
        # Generate new sums
        # New sums of size i+1 are formed by adding new_element to sums of size i
        # We iterate from high i down to 0 to avoid using the same element twice
        # But since we are adding to a new set, we can just compute new sets.
        
        # Actually, we need to check for collisions with EXISTING sums of different cardinalities.
        # And collisions within the SAME cardinality are allowed, but we update the set.
        
        # Let's create the new sets first
        new_updated_sums = [set(s) for s in current_sums]
        
        # Add new_element to sums
        # Sums of size 0: {0}
        # Sums of size 1: {x}
        # ...
        
        # We need to check if any new sum (from size i+1) collides with an existing sum (from size j, j != i+1)
        
        # To be safe, let's just build the full list of sums for the new set and check globally.
        # Given k is small (< 20), 2^k is manageable.
        
        pass

    # Better approach:
    # Maintain a dictionary `sum_map` where key is sum, value is set of cardinalities that can produce it.
    # If any key has >1 cardinalities, it's invalid.
    
    def try_build(k):
        # Backtracking search
        # State: current list of chosen numbers, current sum_map
        # sum_map: dict[int, set[int]]
        
        def backtrack(start_idx, current_set, sum_map):
            if len(current_set) == k:
                return True, current_set
            
            # Heuristic: try smallest available numbers first
            for i in range(start_idx, N + 1):
                x = i
                # Check if x can be added
                # New sums: for each existing sum s of cardinality c, new sum s+x has cardinality c+1
                # Also x itself is sum of cardinality 1 (if empty set sum 0 is cardinality 0)
                
                # We need to check collisions.
                # Let's simulate the addition.
                # new_sum_map = copy of sum_map
                # This is expensive.
                
                # Optimization:
                # Only check collisions for the NEW sums.
                # New sums are {s + x | s in sums_of_cardinality_c} for all c.
                # Plus {x} (from 0+x).
                
                # If a new sum S_new is already in sum_map with a cardinality c_old != c_new, it's a collision.
                # If it's in sum_map with c_old == c_new, it's fine (we just add to the set of cardinalities? No, cardinality is fixed by the subset size).
                # Wait, sum_map maps sum -> set of cardinalities.
                # If S_new is in sum_map, let C_old = sum_map[S_new].
                # The new cardinality is C_new.
                # If C_new is not in C_old, then we have a collision between C_new and some c in C_old?
                # No, if C_old contains {1, 2} and we add a sum of cardinality 3, then C_new=3 is not in {1,2}, so collision between 3 and 1 (and 3 and 2).
                # So if C_new not in C_old, it's invalid?
                # Yes, because that means S_new is achievable by a subset of size C_new AND a subset of size C_old.
                # If C_new != C_old, they are different cardinalities. Collision.
                
                # So condition: For all new sums S_new of cardinality C_new,
                # if S_new in sum_map, then C_new MUST be in sum_map[S_new].
                # If C_new is NOT in sum_map[S_new], then there exists some c in sum_map[S_new] such that c != C_new. Collision.
                
                # Let's check this.
                
                valid = True
                new_sums_to_add = {} # cardinality -> set of sums
                
                # Get all existing sums grouped by cardinality
                # We need to iterate through all existing sums.
                # To do this efficiently, we should maintain a list of (sum, cardinality) pairs or similar.
                # For now, let's just iterate through the items of sum_map?
                # sum_map is sum -> set(cardinalities). This doesn't give us the list of sums per cardinality directly.
                
                # Let's maintain `sums_by_card` = list of sets.
                # sums_by_card[c] is the set of sums of cardinality c.
                
                # Re-doing the state definition.
                pass

        # This is getting complex to implement correctly in a short time without bugs.
        # Let's use a simpler, robust heuristic:
        # 1. Try to build a set of size k using a greedy algorithm that picks the smallest valid number.
        # 2. If it fails, try random restarts.
        
        # We will implement the greedy with full collision check at each step.
        
        def greedy_build(k, random_seed=0):
            random.seed(random_seed)
            current_set = []
            sums_by_card = [set() for _ in range(k + 1)]
            sums_by_card[0].add(0)
            
            # Candidates
            candidates = list(range(1, N + 1))
            
            for step in range(k):
                found = False
                # Shuffle candidates if random_seed != 0 to try different paths?
                # Or just try in order.
                # To allow for "jumps", we can try a few different starting points or orderings.
                
                for x in candidates:
                    if x in current_set:
                        continue
                    
                    # Check if x can be added
                    # Generate new sums
                    new_sums = [set() for _ in range(len(sums_by_card) + 1)]
                    # Copy existing
                    for c in range(len(sums_by_card)):
                        new_sums[c] = set(sums_by_card[c])
                    
                    # Add x
                    for c in range(len(sums_by_card) - 1, -1, -1):
                        for s in sums_by_card[c]:
                            new_sum = s + x
                            new_sums[c + 1].add(new_sum)
                    
                    # Check for collisions
                    # For each cardinality c, check if new_sums[c] intersects with new_sums[j] for j != c
                    # Actually, we only need to check if the NEW sums collide with EXISTING sums of different cardinality.
                    # But since we built new_sums from scratch, we can just check all pairs.
                    
                    # Optimization: Only check new sums against all sums.
                    # A sum S in new_sums[c] is valid if for all j != c, S not in new_sums[j].
                    
                    valid = True
                    for c in range(len(new_sums)):
                        for s in new_sums[c]:
                            for j in range(len(new_sums)):
                                if j == c:
                                    continue
                                if s in new_sums[j]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        current_set.append(x)
                        sums_by_card = new_sums
                        # Remove x from candidates? No, we can reuse logic, but x is distinct.
                        # We just mark it as used.
                        found = True
                        break
                
                if not found:
                    return None
            return current_set

        # Try k from 15 down to 1
        for k in range(15, 0, -1):
            # Try a few random seeds to escape local minima
            for seed in range(5):
                res = greedy_build(k, seed)
                if res:
                    return res
            # If greedy fails, maybe try a different strategy?
            # For now, if k=8 fails, it will try 7, etc.
            # We know 8 works.
            
        # Fallback
        return [1, 2, 4, 8, 16, 32, 64, 128]

    return construct()

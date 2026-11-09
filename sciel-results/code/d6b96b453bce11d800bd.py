import math
import random

def construct():
    max_mod = 128
    candidates = list(range(2, max_mod + 1))
    
    # Precompute gcds to speed up conflict checks
    # gcd_matrix[m1][m2]
    # Or just compute on the fly, math.gcd is fast.
    
    def is_compatible(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def find_assignment(moduli):
        """
        Attempts to find a valid residue assignment for the given list of moduli.
        Returns a dict {m: r} if successful, else None.
        Uses backtracking.
        """
        if not moduli:
            return {}
        
        # Sort moduli to optimize backtracking.
        # Heuristic: Moduli with many non-coprime relationships should be assigned first?
        # Or moduli with small value (fewer residues to try)?
        # Let's try sorting by number of non-coprime neighbors in the set, descending.
        
        def get_degree(m):
            deg = 0
            for m2 in moduli:
                if m != m2 and math.gcd(m, m2) > 1:
                    deg += 1
            return deg
            
        sorted_moduli = sorted(moduli, key=get_degree, reverse=True)
        
        assignment = {}
        
        def backtrack(idx):
            if idx == len(sorted_moduli):
                return True
            
            m = sorted_moduli[idx]
            # Try residues. 
            # Optimization: We can try residues that are "rare" or random?
            # For deterministic correctness, we just iterate.
            # To make it faster, we can randomize the order of residues in some runs?
            # For now, standard order.
            
            for r in range(m):
                # Check against all already assigned
                valid = True
                for m2, r2 in assignment.items():
                    if not is_compatible(r, m, r2, m2):
                        valid = False
                        break
                
                if valid:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    best_size = 0
    best_moduli = []
    best_assignment = {}
    
    # Randomized Greedy with Restarts
    # Number of restarts depends on time. Let's do 200 restarts.
    # Each restart iterates 127 moduli.
    # Feasibility check is called 127 times per restart.
    # Total 25k feasibility checks.
    # If each takes ~1ms, that's 25s. Too slow.
    # We need to optimize the feasibility check or reduce restarts.
    
    # Optimization: 
    # In the greedy step, we don't need to re-solve from scratch.
    # We can maintain the current assignment.
    # When trying to add m, we check if there is an r for m compatible with current assignment.
    # If yes, we add it.
    # If no, we skip it.
    # This is the "Greedy without Re-optimization" approach.
    # It is very fast.
    # Does it find optimal? No, but with random restarts (shuffling order), it explores different local optima.
    
    # Let's use the Fast Greedy (no re-optimization of existing residues) with many restarts.
    
    def fast_greedy(order):
        current_moduli = []
        current_assignment = {}
        
        for m in order:
            # Find a residue r for m compatible with current_assignment
            found = False
            # Randomize residue order? No, keep deterministic for reproducibility of a specific shuffle, 
            # but the shuffle itself provides the randomness.
            # Actually, trying residues in random order might help if we want to explore more, 
            # but for speed, 0..m-1 is fine.
            
            for r in range(m):
                valid = True
                for m2, r2 in current_assignment.items():
                    if not is_compatible(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    current_assignment[m] = r
                    current_moduli.append(m)
                    found = True
                    break
            # If not found, skip m
            
        return current_moduli, current_assignment

    # Run multiple restarts
    import time
    start_time = time.time()
    num_restarts = 1000 # Adjust based on time limit
    
    for _ in range(num_restarts):
        if time.time() - start_time > 50:
            break
            
        random_order = candidates[:]
        random.shuffle(random_order)
        
        moduli, assignment = fast_greedy(random_order)
        
        if len(moduli) > best_size:
            best_size = len(moduli)
            best_moduli = moduli
            best_assignment = assignment
            # Optional: Try to improve best_moduli further?
            # For now, just keep it.

    # Convert to required format
    result = [[best_assignment[m], m] for m in best_moduli]
    return result

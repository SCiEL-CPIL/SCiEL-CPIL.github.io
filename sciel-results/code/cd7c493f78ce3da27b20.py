"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety margin: 85s out of 90s
    deadline = _START_TIME + 85.0
    
    # Candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # Precompute GCDs for all pairs of moduli to speed up checks
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    n_mods = len(moduli)
    gcd_matrix = [[0]*n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = _gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Heuristic ordering for backtracking:
    # We want to place "constrained" moduli first.
    # A modulus is more constrained if it shares factors with many other moduli.
    # Let's score each modulus by the sum of gcds with all other moduli.
    # Higher sum = more potential conflicts = should be placed earlier to prune the tree.
    scores = [0] * n_mods
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j:
                scores[i] += gcd_matrix[i][j]
                
    # Sort indices by score descending
    order = sorted(range(n_mods), key=lambda i: -scores[i])
    
    best_solution = []
    best_size = 0
    node_limit = 10000 # Limit nodes per backtracking run to avoid hanging on one path
    
    # We will run multiple randomized backtracking sessions
    # Each session picks a random subset of the order? No, we should try to find the MAX set.
    # Backtracking on the full set of 127 moduli is too big for simple recursion without strong pruning.
    # However, we can use a "greedy backtracking" hybrid:
    # 1. Do a greedy pass to get a good initial solution.
    # 2. Use local search (simulated annealing) to improve it.
    
    # Let's stick to a robust Local Search approach which is often better for these "max independent set" like problems on graphs.
    # The graph: Nodes are (modulus, residue) pairs? No, that's 10000+ nodes.
    # Nodes are just Moduli? No, the residue matters.
    
    # Let's try a different angle:
    # The problem is equivalent to finding a large set of moduli M and assigning residues R such that for all m1, m2 in M, 
    # the classes are disjoint.
    
    # Strategy:
    # 1. Generate a random valid assignment using backtracking with a small depth limit (e.g., try to place first 30 moduli).
    # 2. Use Local Search to add more moduli or swap residues to allow more moduli.
    
    def get_initial_solution():
        """Greedy construction with random residue shuffling."""
        current = []
        used_moduli = set()
        # Use the heuristic order
        for idx in order:
            m = moduli[idx]
            if m in used_moduli:
                continue
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                # Check against all existing
                valid = True
                for (r_e, m_e) in current:
                    i_e = moduli.index(m_e) # This is slow, let's optimize
                    # We need index mapping
                    if not _disjoint(r, m, r_e, m_e):
                        valid = False
                        break
                if valid:
                    current.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
        return current

    # Optimize _disjoint check by using precomputed gcd and indices
    # Map modulus value to index in moduli list
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    def is_valid_pair(r1, m1, r2, m2):
        i1 = mod_to_idx[m1]
        i2 = mod_to_idx[m2]
        g = gcd_matrix[i1][i2]
        if g == 1:
            return False
        return (r2 - r1) % g != 0

    def can_add(r_new, m_new, existing):
        for (r, m) in existing:
            if not is_valid_pair(r_new, m_new, r, m):
                return False
        return True

    best_sol = []
    
    # Run multiple local search instances
    # Each instance:
    # 1. Start with a random greedy solution.
    # 2. Perform hill climbing / simulated annealing.
    #    Moves:
    #    - Add a new modulus/residue if valid.
    #    - Remove a modulus/residue.
    #    - Change residue of an existing modulus.
    
    for restart in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        current = get_initial_solution()
        current_size = len(current)
        
        if current_size > best_size:
            best_size = current_size
            best_sol = [x[:] for x in current]
            
        # Local Search Phase
        # We want to maximize size.
        # If we are stuck, we can try to "swap" residues to free up a slot for a new modulus.
        
        # Simple Hill Climbing: Try to add any missing modulus.
        # If no single addition works, try a pair of moves (swap residue + add).
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add a new modulus
            missing_moduli = [m for m in moduli if m not in [x[1] for x in current]]
            random.shuffle(missing_moduli)
            
            added = False
            for m in missing_moduli:
                if time.time() > deadline:
                    break
                # Try all residues
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if can_add(r, m, current):
                        current.append([r, m])
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_sol = [x[:] for x in current]
                        added = True
                        break
                if added:
                    break
            
            if added:
                improved = True
                continue # Loop again to add more
            
            # 2. If no direct addition, try to modify existing residues to enable an addition.
            # Pick a missing modulus m_target.
            if missing_moduli:
                m_target = missing_moduli[0] # Or random
                # Find which existing moduli conflict with m_target for ALL residues?
                # Actually, for a specific m_target, there might be no valid residue.
                # We need to change residues of existing moduli to make one valid.
                
                # Let's try a random swap: Pick a random existing modulus m_old, change its residue.
                # Does this help? Only if it was blocking m_target.
                
                # Heuristic: Identify the "bottleneck" modulus.
                # For m_target, calculate for each existing modulus m_i, the set of residues r_target that are invalid due to m_i.
                # If the union of these invalid sets covers [0, m_target), we are stuck.
                # We need to change m_i's residue to a value that invalidates fewer r_target options?
                # Or rather, makes a specific r_target valid.
                
                # Let's just do random perturbation:
                # Pick a random existing pair, change its residue to a random valid one (if any), 
                # then try to add m_target.
                
                # To avoid infinite loops, limit perturbation steps.
                for _ in range(10): # 10 perturbation attempts
                    if time.time() > deadline:
                        break
                    
                    # Pick a random existing element
                    idx = random.randint(0, len(current)-1)
                    r_old, m_old = current[idx]
                    
                    # Generate a new random residue for m_old
                    # Ensure it doesn't conflict with OTHER existing elements (except m_target which we haven't added yet)
                    # Actually, we just need it to be valid with the rest of the current set (excluding itself).
                    new_residues = [r for r in range(m_old) if r != r_old]
                    random.shuffle(new_residues)
                    
                    swapped = False
                    for r_new in new_residues:
                        # Check if r_new is valid with all others in current
                        valid = True
                        for j, (r_j, m_j) in enumerate(current):
                            if j == idx:
                                continue
                            if not is_valid_pair(r_new, m_old, r_j, m_j):
                                valid = False
                                break
                        
                        if valid:
                            # Perform swap
                            current[idx] = [r_new, m_old]
                            swapped = True
                            
                            # Now try to add m_target
                            added = False
                            residues_t = list(range(m_target))
                            random.shuffle(residues_t)
                            for r_t in residues_t:
                                if can_add(r_t, m_target, current):
                                    current.append([r_t, m_target])
                                    current_size += 1
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_sol = [x[:] for x in current]
                                    added = True
                                    break
                            
                            if added:
                                improved = True
                                break # Break perturbation loop, go back to main while loop
                            
                            # If not added, revert? Or keep the swap if it didn't hurt?
                            # If we keep the swap, we might get stuck in a local optimum of the "current" configuration.
                            # Let's revert to maintain the "hill climbing" property (only accept if size increases).
                            # If size doesn't increase, revert.
                            current[idx] = [r_old, m_old]
                            break # Try next perturbation candidate? Or break entirely?
                            # Let's break the inner residue loop and try a different swap candidate.
                            
                    if improved:
                        break
                    if swapped:
                        # If we swapped but didn't add, we reverted. 
                        # If we didn't swap, try next perturbation.
                        pass

    if not best_sol:
        return [[1, 2]]
        
    return best_sol

# EVOLVE-BLOCK-END

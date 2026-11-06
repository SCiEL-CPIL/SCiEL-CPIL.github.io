"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    n_candidates = len(candidates)
    
    # Precompute 1/m as float for speed
    inv_float = {m: 1.0 / m for m in candidates}
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for i, m1 in enumerate(candidates):
        for m2 in candidates[i:]:
            g = gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            gcd_table[(m2, m1)] = g
    
    # Build the conflict structure
    # For each modulus m, and each residue r, we need to know which other (m', r') pairs it conflicts with
    # Two pairs (m, r) and (m', r') conflict if:
    #   1. m == m' (same modulus), OR
    #   2. r % gcd(m, m') == r' % gcd(m, m')
    
    # Key structural insight: 
    # For a fixed modulus m, the residue r determines which "congruence classes" it occupies
    # for each other modulus m'. Specifically, (m, r) occupies class r % gcd(m, m') in modulus m'.
    
    # New approach: Formulate as maximum weight independent set on a special graph
    # where nodes are (m, r) pairs, and use branch-and-bound with strong bounds.
    
    # But first, let's use a different structural approach:
    # Model this as a constraint satisfaction problem and use arc consistency
    # to prune the search space, then use a weighted constraint optimization.
    
    # Step 1: For each modulus m, determine which residues are "feasible"
    # given that we want to maximize sum of 1/m.
    
    # Step 2: Use a greedy + local search approach with a different representation.
    
    # Actually, let me try a completely different approach:
    # Use the fact that the problem has a special structure related to Chinese Remainder Theorem.
    
    # For two moduli m1, m2 with gcd d = gcd(m1, m2):
    # (m1, r1) and (m2, r2) are disjoint iff r1 != r2 (mod d)
    # This means they occupy different "slots" in the d-way partition.
    
    # New structural approach: Think of this as a graph coloring / packing problem.
    # Each modulus m has m "slots" (residues 0 to m-1).
    # For each pair of moduli (m1, m2) with gcd d, we need to assign residues
    # such that they don't share the same slot in the d-way partition.
    
    # Let me try a different formulation:
    # For each modulus m, define a "signature" vector of length LCM(20,...,200)
    # but that's too large.
    
    # Instead, use a practical approach:
    # 1. Start with a good initial solution using a smarter greedy
    # 2. Use a different local search with 2-opt style moves
    # 3. Add a constraint propagation step
    
    def is_disjoint(m_new, r_new, existing):
        """Check if (m_new, r_new) is disjoint from all in existing list of (m, r)."""
        for (m_e, r_e) in existing:
            g = gcd_table.get((m_new, m_e), gcd(m_new, m_e))
            if r_new % g == r_e % g:
                return False
        return True
    
    def find_valid_residue(m, existing):
        """Find any valid residue for m given existing assignments."""
        for r in range(m):
            if is_disjoint(m, r, existing):
                return r
        return None
    
    # Phase 1: Smarter greedy initialization
    # Sort moduli by 1/m descending, but also consider "flexibility"
    # A modulus m is more flexible if it has fewer conflicts with already-selected moduli
    
    def greedy_init_v2():
        selected = []
        # Try multiple orderings and pick the best
        best_selected = []
        best_score = 0
        
        orderings = [
            sorted(candidates),  # ascending m
            sorted(candidates, reverse=True),  # descending m
            sorted(candidates, key=lambda m: -inv_float[m]),  # by weight
        ]
        
        # Also try a "conflict-aware" ordering
        for ordering in orderings:
            if time.time() > deadline:
                break
            selected = []
            for m in ordering:
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append((m, r))
            score = sum(inv_float[m] for m, r in selected)
            if score > best_score:
                best_score = score
                best_selected = selected[:]
        
        return best_selected, best_score
    
    best, best_score = greedy_init_v2()
    
    # Phase 2: Constraint propagation + local search
    random.seed(42)
    
    def compute_score(state):
        return sum(inv_float[m] for m, r in state)
    
    def try_add(state, m, r):
        """Try to add (m, r) to state, return new state if valid, else None."""
        used_mods = set(x[0] for x in state)
        if m in used_mods:
            return None
        for (m_e, r_e) in state:
            g = gcd_table.get((m, m_e), gcd(m, m_e))
            if r % g == r_e % g:
                return None
        return state + [(m, r)]
    
    def try_remove(state, idx):
        return state[:idx] + state[idx+1:]
    
    # New local search: Use a different move set
    # - Add a modulus with a specific residue
    # - Remove a modulus
    # - Change residue of a modulus
    # - Swap: remove one, add another
    # - 2-opt: remove two, add two
    
    def local_search(state, score, time_limit):
        """Perform local search with multiple move types."""
        import math
        T = 0.3
        T_min = 0.001
        cooling = 0.999
        best_state = state
        best_score = score
        no_improve_count = 0
        
        while T > T_min and time.time() < time_limit:
            improved = False
            
            # Try each move type
            moves = []
            
            # Add moves
            used_mods = set(m for m, r in state)
            available = [m for m in candidates if m not in used_mods]
            if available:
                # Try adding each available modulus with a random residue
                for m_new in available[:30]:  # Limit to top 30 by weight
                    for attempt in range(5):
                        r_new = random.randint(0, m_new - 1)
                        new_state = try_add(state, m_new, r_new)
                        if new_state:
                            new_score = compute_score(new_state)
                            moves.append(('add', new_state, new_score, m_new, r_new))
            
            # Remove moves
            if len(state) > 1:
                for idx in random.sample(range(len(state)), min(10, len(state))):
                    new_state = try_remove(state, idx)
                    new_score = compute_score(new_state)
                    moves.append(('remove', new_state, new_score, state[idx][0], state[idx][1]))
            
            # Change residue moves
            for idx in random.sample(range(len(state)), min(10, len(state))):
                m_old, r_old = state[idx]
                for attempt in range(5):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd_table.get((m_old, m_e), gcd(m_old, m_e))
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        moves.append(('change_r', new_state, new_score, m_old, r_new))
            
            # 2-opt moves: remove two, add two
            if len(state) >= 2:
                idx1, idx2 = random.sample(range(len(state)), 2)
                m1, r1 = state[idx1]
                m2, r2 = state[idx2]
                new_state_without = state[:min(idx1,idx2)] + state[min(idx1,idx2)+1:max(idx1,idx2)] + state[max(idx1,idx2)+1:]
                used_mods = set(m for m, r in new_state_without)
                available = [m for m in candidates if m not in used_mods]
                if len(available) >= 2:
                    m_new1, m_new2 = random.sample(available[:20], 2)
                    for r_new1 in range(min(m_new1, 5)):
                        for r_new2 in range(min(m_new2, 5)):
                            new_state = try_add(new_state_without, m_new1, r_new1)
                            if new_state:
                                new_state = try_add(new_state, m_new2, r_new2)
                                if new_state:
                                    new_score = compute_score(new_state)
                                    moves.append(('2opt', new_state, new_score, (m1, r1), (m2, r2)))
            
            # Accept the best move or a random move with SA probability
            if moves:
                best_move = max(moves, key=lambda x: x[2])
                delta = best_move[2] - score
                if delta > 0 or random.random() < math.exp(delta / T):
                    state = best_move[1]
                    score = best_move[2]
                    improved = True
                    if score > best_score:
                        best_state = state[:]
                        best_score = score
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            T *= cooling
            
            # Early stop if no improvement for many iterations
            if no_improve_count > 100:
                break
        
        return best_state, best_score
    
    # Run local search with time budget
    if time.time() < deadline - 30:
        best, best_score = local_search(best, best_score, deadline - 30)
    
    # Phase 3: Try to add more moduli using a constraint propagation approach
    used_mods = set(m for m, r in best)
    remaining = [m for m in candidates if m not in used_mods]
    remaining.sort()  # By m ascending (higher weight first)
    
    if remaining and time.time() < deadline - 10:
        # For each remaining modulus, try to find a valid residue
        # Use a more systematic approach: try all residues, not just the first valid one
        for m in remaining[:50]:  # Try top 50 remaining
            if time.time() > deadline - 10:
                break
            # Try all residues to find the best one (though score is same, different residues may enable future additions)
            valid_residues = []
            for r in range(m):
                if is_disjoint(m, r, best):
                    valid_residues.append(r)
            
            if valid_residues:
                # Try each valid residue and see which one allows the most future additions
                best_r = None
                best_future_count = -1
                for r in valid_residues[:10]:  # Limit to 10 residues
                    test_state = best + [(m, r)]
                    # Count how many additional moduli can be added
                    future_count = 0
                    for m2 in remaining:
                        if m2 == m:
                            continue
                        if find_valid_residue(m2, test_state) is not None:
                            future_count += 1
                    if future_count > best_future_count:
                        best_future_count = future_count
                        best_r = r
                
                if best_r is not None:
                    new_state = best + [(m, best_r)]
                    new_score = best_score + inv_float[m]
                    if new_score > best_score:
                        best, best_score = new_state, new_score
    
    # Final local search pass
    if time.time() < deadline - 5:
        best, best_score = local_search(best, best_score, deadline - 5)
    
    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END

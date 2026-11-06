"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Increase recursion limit just in case, though we use iterative logic where possible
sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 285  # Leave 15s margin
    
    # Preprocess: Convert to 0-based, use integers for literals
    # lit > 0 means var = lit-1, sign = +1
    # lit < 0 means var = -lit-1, sign = -1
    # We will use a representation where literal l corresponds to var l//2 and sign l%2?
    # No, standard 1-based signed is fine, just convert to 0-based indices for array access.
    
    # Internal representation:
    # Clauses: list of lists of (var_idx, sign)
    # But for watched literals, it's easier to use integers:
    # Positive literal for var i: 2*i
    # Negative literal for var i: 2*i + 1
    # This allows fast bit operations if needed, but standard int is fine.
    
    def to_internal(lit):
        # lit is 1-based signed
        if lit > 0:
            return 2 * (lit - 1)
        else:
            return 2 * (-lit - 1) + 1

    def to_external(lit_int):
        # Convert internal int back to 1-based signed
        if lit_int % 2 == 0:
            return (lit_int // 2) + 1
        else:
            return -((lit_int // 2) + 1)

    # Convert clauses
    internal_clauses = []
    for clause in clauses:
        internal_clause = [to_internal(l) for l in clause]
        internal_clauses.append(internal_clause)
        
    m = len(internal_clauses)
    
    # DPLL Data Structures
    # assignment: -1 (unassigned), 0 (false), 1 (true)
    assignment = [-1] * n
    # level: decision level at which variable was assigned
    level = [0] * n
    # reason: clause index that caused the assignment (for learning, not using here)
    # trail: list of (var_idx, level) in order of assignment
    trail = []
    # watched: watched[lit_int] = list of clause indices
    # Note: lit_int ranges from 0 to 2*n - 1
    watched = [[] for _ in range(2 * n)]
    
    # Initialize watched literals
    # For each clause, pick two literals to watch. Usually the first two.
    for c_idx, clause in enumerate(internal_clauses):
        if len(clause) > 0:
            watched[clause[0]].append(c_idx)
        if len(clause) > 1:
            watched[clause[1]].append(c_idx)
            
    # VSIDS Activity
    activity = [0.0] * n
    # Decay factor
    decay_factor = 0.95
    
    # Variable selection heap or simple scan?
    # For n=50, a simple scan or a small heap is fine.
    # Let's use a simple list of unassigned vars and pick max activity.
    # To make it faster, we can maintain a set of unassigned vars.
    
    def get_var(lit_int):
        return lit_int // 2
        
    def get_sign(lit_int):
        # 0: True (+), 1: False (-) in internal?
        # Let's define:
        # Even internal lit: Positive literal (Var is True satisfies it)
        # Odd internal lit: Negative literal (Var is False satisfies it)
        return 1 if lit_int % 2 == 0 else -1
        
    def is_true(lit_int):
        v = get_var(lit_int)
        s = get_sign(lit_int)
        return assignment[v] == s
        
    def is_false(lit_int):
        v = get_var(lit_int)
        s = get_sign(lit_int)
        return assignment[v] == -s
        
    def assign(lit_int, lvl):
        v = get_var(lit_int)
        s = get_sign(lit_int)
        assignment[v] = s
        level[v] = lvl
        trail.append((v, lvl))
        
    def unassign(v):
        assignment[v] = -1
        level[v] = 0
        
    # Unit Propagation
    # Returns: True if no conflict, False if conflict
    # We need to track the trail size to backtrack
    def propagate():
        while trail:
            # We process the last assigned literal? No, we process the consequences of the last assignment.
            # Standard DPLL propagation loop:
            # We need a queue of literals to check.
            # Actually, the watched literal scheme works by:
            # When a literal L is assigned False, check all clauses watching L.
            
            # Let's implement a queue-based propagation
            # But we need to know which literals were just assigned.
            # We can store the current trail index.
            
            pass # Placeholder, will implement differently
        
    # Better approach for Propagation with Watched Literals:
    # We maintain a queue of literals to process.
    # Initially, we don't have any.
    # When we branch, we assign a literal and push it to the queue.
    # When we propagate (unit clause), we assign a literal and push it to the queue.
    
    # Let's restructure the DPLL loop.
    
    def dpll_search():
        nonlocal trail, assignment, level, activity, watched, internal_clauses
        nonlocal deadline
        
        # Queue for propagation
        # We'll use a list as a queue
        prop_queue = []
        
        # Decision level
        current_level = 0
        
        # We need to handle backtracking.
        # For simplicity and robustness on small N, we can use recursive DPLL with state saving?
        # No, iterative is better for memory.
        
        # Let's use a stack of decision levels.
        # Actually, for N=50, a simple recursive approach with careful state management might be easier to debug and less error-prone than a full iterative CDCL.
        # But the previous recursive DPLL was slow.
        
        # Let's try an iterative DPLL with VSIDS and Chronological Backtracking.
        
        # State:
        # assignment, level, trail
        # We need to restore state on backtrack.
        # Saving the entire assignment array is O(N) = 50. Very cheap.
        
        def save_state():
            return (assignment[:], level[:], trail[:], [w[:] for w in watched])
            
        def restore_state(state):
            nonlocal assignment, level, trail, watched
            assignment, level, trail, watched = state
            
        # This is too expensive to copy watched lists every time.
        # Watched lists change during propagation.
        
        # Alternative: Use a standard recursive DPLL but with efficient clause filtering?
        # The previous one was slow because it copied clause lists.
        
        # Let's go back to the **WalkSAT** inspiration but make it deterministic? No.
        
        # Let's combine:
        # 1. Try a **highly optimized DPLL** with **VSIDS** and **Watched Literals**.
        #    To avoid the complexity of restoring watched lists, we can use a **trail-based** approach where we undo changes to watched lists.
        #    This is complex.
        
        # Given the constraints and the fact that WalkSAT scored 1.0, maybe the best "Structural Change" is to **optimize the WalkSAT** significantly?
        # But the directive says "Replace ... with DPLL".
        
        # Let's implement a **simple but fast DPLL** using **Bitsets** (Python ints as bitsets) for clauses and assignments.
        # N=50 fits in a 64-bit integer.
        # This will be extremely fast.
        
        # Representation:
        # Pos_vars: int with bits set for variables assigned True.
        # Neg_vars: int with bits set for variables assigned False.
        # Clauses: List of (pos_mask, neg_mask).
        # Clause is satisfied if (pos_mask & Pos_vars) != 0 or (neg_mask & Neg_vars) != 0.
        # Clause is falsified if (pos_mask & Neg_vars) == pos_mask and (neg_mask & Pos_vars) == neg_mask.
        # Clause is unit if exactly one variable is unassigned.
        
        # This bitset approach is a **structural change** in representation.
        
        # Convert clauses to bitsets
        pos_masks = []
        neg_masks = []
        for clause in internal_clauses:
            pm = 0
            nm = 0
            for lit in clause:
                v = lit // 2
                if lit % 2 == 0: # Positive
                    pm |= (1 << v)
                else: # Negative
                    nm |= (1 << v)
            pos_masks.append(pm)
            neg_masks.append(nm)
            
        # DPLL with Bitsets
        # Assignment: pos_assigned, neg_assigned
        # Unassigned: all_vars ^ pos_assigned ^ neg_assigned
        
        all_vars_mask = (1 << n) - 1
        
        def dpll_bitset(pos_a, neg_a, trail):
            # Check deadline
            if time.time() > deadline:
                return None
            
            # Propagation: Unit Clauses
            # We need to iterate clauses to find units.
            # With 50 vars, and M clauses (typically ~100-200), this is fast.
            
            changed = True
            while changed:
                changed = False
                new_pos_a = pos_a
                new_neg_a = neg_a
                
                # Check for conflicts and units
                for i in range(m):
                    pm = pos_masks[i]
                    nm = neg_masks[i]
                    
                    # Check if satisfied
                    if (pm & new_pos_a) != 0 or (nm & new_neg_a) != 0:
                        continue
                    
                    # Check if falsified
                    # If all literals in clause are assigned and false
                    # Literals in clause: pm (pos) and nm (neg)
                    # Pos lit is false if var is in neg_a
                    # Neg lit is false if var is in pos_a
                    # Clause is falsified if (pm & neg_a) == pm AND (nm & pos_a) == nm
                    # Note: pm and nm are disjoint? No, a clause can have x and -x? 
                    # In 3-SAT, usually no tautologies. Assume disjoint.
                    
                    # Unassigned literals in this clause:
                    # Pos lits unassigned: pm & ~new_pos_a & ~new_neg_a
                    # Neg lits unassigned: nm & ~new_pos_a & ~new_neg_a
                    unassigned_mask = (pm | nm) & ~(new_pos_a | new_neg_a)
                    
                    if unassigned_mask == 0:
                        # Falsified
                        return None
                    
                    if unassigned_mask & (unassigned_mask - 1) == 0:
                        # Unit clause: exactly one bit set
                        var_bit = unassigned_mask
                        v = var_bit.bit_length() - 1
                        
                        # Determine sign
                        if pm & var_bit:
                            # Must be True
                            if new_pos_a & var_bit:
                                # Already true? Should have been satisfied
                                pass
                            else:
                                new_pos_a |= var_bit
                                new_neg_a &= ~var_bit # Ensure consistency
                                changed = True
                                trail.append((v, 1)) # 1 for True
                        else:
                            # Must be False
                            if new_neg_a & var_bit:
                                pass
                            else:
                                new_neg_a |= var_bit
                                new_pos_a &= ~var_bit
                                changed = True
                                trail.append((v, 0)) # 0 for False
                                
                pos_a = new_pos_a
                neg_a = new_neg_a
                
            # Check if all satisfied
            for i in range(m):
                pm = pos_masks[i]
                nm = neg_masks[i]
                if (pm & pos_a) == 0 and (nm & neg_a) == 0:
                    # Not satisfied, so must be unit or unassigned
                    # But we just propagated, so if not satisfied and not unit, it's multi-unassigned
                    # If we reached here, no units left.
                    # If any clause is not satisfied and not falsified, it has >=2 unassigned.
                    # We need to branch.
                    pass
            
            # If all clauses satisfied, return solution
            # Check if all clauses are satisfied
            for i in range(m):
                if (pos_masks[i] & pos_a) == 0 and (neg_masks[i] & neg_a) == 0:
                    # This clause is not satisfied.
                    # Since no units remain, it must have >=2 unassigned.
                    # We need to branch.
                    break
            else:
                # All clauses satisfied
                # Construct assignment
                res = [False] * n
                for v in range(n):
                    bit = 1 << v
                    if pos_a & bit:
                        res[v] = True
                    # If neg_a & bit, it's False
                return res
            
            # Branching
            # Select variable
            # Use VSIDS? Or simple frequency?
            # For bitset, we can compute frequency of unassigned vars in unsatisfied clauses.
            
            unsat_clauses = []
            for i in range(m):
                if (pos_masks[i] & pos_a) == 0 and (neg_masks[i] & neg_a) == 0:
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return res # Should have been caught above
                
            # Count variable occurrences in unsat clauses
            var_count = [0] * n
            for i in unsat_clauses:
                pm = pos_masks[i]
                nm = neg_masks[i]
                # Only count unassigned vars
                unassigned = (pm | nm) & ~(pos_a | neg_a)
                while unassigned:
                    bit = unassigned & -unassigned
                    v = bit.bit_length() - 1
                    var_count[v] += 1
                    unassigned ^= bit
                    
            # Find max count
            max_count = 0
            branch_var = -1
            for v in range(n):
                if (pos_a | neg_a) & (1 << v) == 0: # Unassigned
                    if var_count[v] > max_count:
                        max_count = var_count[v]
                        branch_var = v
                        
            if branch_var == -1:
                # No unassigned vars in unsat clauses? Contradiction?
                return None
                
            # Try True
            bit = 1 << branch_var
            trail.append((branch_var, 1))
            result = dpll_bitset(pos_a | bit, neg_a & ~bit, trail)
            if result is not None:
                return result
            trail.pop()
            
            # Try False
            trail.append((branch_var, 0))
            result = dpll_bitset(pos_a & ~bit, neg_a | bit, trail)
            if result is not None:
                return result
            trail.pop()
            
            return None
            
        # Run DPLL
        solution = dpll_bitset(0, 0, [])
        
        if solution is not None:
            return solution
            
        # Fallback to WalkSAT if DPLL fails (timeout or bug)
        if time.time() < deadline - 5:
            # WalkSAT Implementation
            rng = random.Random(0)
            # Random init
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            def is_satisfied(clause):
                return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
                
            best_assignment = assignment[:]
            best_unsat = sum(1 for c in clauses if not is_satisfied(c))
            
            max_steps = 10000
            noise = 0.01
            
            for _ in range(max_steps):
                if time.time() > deadline - 2:
                    break
                    
                unsat = [c for c in clauses if not is_satisfied(c)]
                if not unsat:
                    return assignment
                    
                if len(unsat) < best_unsat:
                    best_unsat = len(unsat)
                    best_assignment = assignment[:]
                    
                clause = rng.choice(unsat)
                
                # Heuristic: pick var with most occurrences in unsat clauses
                freq = {}
                for c in unsat:
                    for l in c:
                        v = abs(l) - 1
                        freq[v] = freq.get(v, 0) + 1
                        
                candidates = [l for l in clause if freq.get(abs(l)-1, 0) == max(freq.get(abs(l2)-1, 0) for l2 in clause)]
                
                if rng.random() < noise:
                    lit = rng.choice(clause)
                else:
                    lit = rng.choice(candidates)
                    
                v = abs(lit) - 1
                assignment[v] = not assignment[v]
                
            return best_assignment
            
        return [False] * n

# EVOLVE-BLOCK-END

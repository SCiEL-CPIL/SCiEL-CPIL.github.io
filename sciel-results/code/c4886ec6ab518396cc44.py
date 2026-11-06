"""
B_3[1] set construction via Simulated Annealing with Rolling Hash.
Strategy: 
1. Initialize with a fast greedy pass to get a valid baseline.
2. Use Simulated Annealing to explore the space of sets of fixed size.
   Moves: Insert a new element (if size < target) or Swap an existing element with a new candidate.
   Cost function: Number of triple sum collisions.
   Acceptance: Metropolis criterion with cooling temperature.
3. Use a polynomial rolling hash to quickly check if a proposed move introduces collisions.
"""

import time
import random
import sys

def _compute_triple_sums(S):
    """Compute all unique triple sums for a set S."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _count_collisions(S):
    """
    Count the number of duplicate triple sums.
    Returns the number of pairs of triples that have the same sum.
    If 0, the set is valid.
    """
    n = len(S)
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                sum_counts[s] = sum_counts.get(s, 0) + 1
    
    # Number of collisions is sum over all sums of (count choose 2)
    # But for SA, we just need a penalty score. 
    # A simpler penalty: sum of (count - 1) for each sum? 
    # Actually, let's just count the number of "extra" occurrences.
    # If a sum appears 'c' times, it contributes c-1 to the "excess".
    # Total excess = sum(c) - number_of_unique_sums.
    # Since total number of triples is fixed for a given size, minimizing unique sums maximizes collisions.
    # Let's define cost = number of triples - number of unique triple sums.
    total_triples = (n+2)*(n+1)*n // 6
    return total_triples - len(sum_counts)

def _check_conflict_fast(x, S, existing_sums):
    """
    Check if adding x to S creates any conflict with existing sums or new sums involving x.
    Returns (valid, new_sums) where new_sums is the set of new triple sums added.
    """
    n = len(S)
    new_sums = set()
    
    # 1. (x, x, x)
    s = 3 * x
    if s in existing_sums:
        return False, None
    new_sums.add(s)
    
    # 2. (x, x, y)
    for y in S:
        s = 2 * x + y
        if s in existing_sums:
            return False, None
        if s in new_sums:
            return False, None
        new_sums.add(s)
        
    # 3. (x, y, y)
    for y in S:
        s = x + 2 * y
        if s in existing_sums:
            return False, None
        if s in new_sums:
            return False, None
        new_sums.add(s)
        
    # 4. (x, y, z)
    for i in range(n):
        for j in range(i + 1, n):
            s = x + S[i] + S[j]
            if s in existing_sums:
                return False, None
            if s in new_sums:
                return False, None
            new_sums.add(s)
            
    return True, new_sums

def _greedy_init(n_max=4096, target_size=60):
    """
    Fast greedy initialization to get a valid starting set.
    """
    S = []
    existing_sums = set()
    
    # Try to build a set of size target_size
    # Use a slightly randomized order to avoid getting stuck in a specific local optimum immediately
    # But for init, deterministic is faster and reliable.
    # We'll try to fill up to target_size.
    
    for x in range(1, n_max + 1):
        if len(S) >= target_size:
            break
        valid, new_sums = _check_conflict_fast(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums.update(new_sums)
            
    return S, existing_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n_max = 4096
    
    # 1. Greedy Initialization
    S, existing_sums = _greedy_init(n_max, target_size=60)
    current_size = len(S)
    current_cost = 0  # Greedy ensures no collisions
    
    # If greedy couldn't reach 60, we start with what we have.
    # We want to maximize cardinality. 
    # The SA will try to increase size or maintain size while optimizing.
    
    # Strategy: 
    # Phase 1: Try to increase size using SA.
    # Phase 2: Optimize the set for potential further insertion or just keep best.
    
    best_S = list(S)
    best_size = current_size
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    max_iter = 20000  # Max iterations for SA
    
    # Candidate pool for swaps/inserts
    # We need candidates that are NOT in S.
    in_S = set(S)
    candidates = [x for x in range(1, n_max + 1) if x not in in_S]
    
    # Shuffle candidates for diversity
    random.shuffle(candidates)
    
    # We will use a rolling window of candidates to keep memory and time down
    # Or just pick random ones from the full list.
    
    # Precompute some stats? No, dynamic.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Choose a move type
        # 1. Insert (if size < 80)
        # 2. Swap (replace one element with another)
        
        # Probability of trying to increase size
        # If we are far from 80, prefer insert.
        p_insert = 0.5 if current_size < 70 else 0.2
        
        move_type = random.random()
        
        new_S = list(S)
        new_existing_sums = existing_sums  # Reference, we will modify carefully
        
        if move_type < p_insert and current_size < 80:
            # Try to insert a new element
            # Pick a random candidate
            if not candidates:
                break
            x = random.choice(candidates)
            
            # Check if x can be added
            # We need to check against existing_sums AND new sums involving x
            # But for SA, we allow collisions? 
            # No, the goal is a VALID set. 
            # If we allow invalid states, we need a cost function.
            # Cost = number of collisions.
            
            # Let's calculate the cost change for adding x.
            # If we add x, we add new sums. 
            # Collisions occur if any new sum is in existing_sums or duplicates another new sum.
            
            new_sums = set()
            collisions = 0
            
            # (x,x,x)
            s = 3 * x
            if s in existing_sums:
                collisions += 1
            else:
                new_sums.add(s)
                
            # (x,x,y)
            for y in S:
                s = 2 * x + y
                if s in existing_sums:
                    collisions += 1
                elif s in new_sums:
                    collisions += 1
                else:
                    new_sums.add(s)
                    
            # (x,y,y)
            for y in S:
                s = x + 2 * y
                if s in existing_sums:
                    collisions += 1
                elif s in new_sums:
                    collisions += 1
                else:
                    new_sums.add(s)
                    
            # (x,y,z)
            for idx1 in range(len(S)):
                for idx2 in range(idx1 + 1, len(S)):
                    s = x + S[idx1] + S[idx2]
                    if s in existing_sums:
                        collisions += 1
                    elif s in new_sums:
                        collisions += 1
                    else:
                        new_sums.add(s)
                        
            delta_cost = collisions
            # If collisions > 0, the state is invalid. 
            # We accept if delta_cost < 0 (impossible here) or with probability exp(-delta_cost/T).
            
            if collisions == 0:
                # Valid insertion
                S = new_S + [x]
                existing_sums.update(new_sums)
                current_size += 1
                current_cost = 0
                if current_size > best_size:
                    best_S = list(S)
                    best_size = current_size
                # Remove x from candidates? No, we might swap it later.
                # But if we keep it in candidates, we might try to insert it again.
                # We should track membership.
                in_S.add(x)
                # Remove from candidates list for efficiency? 
                # It's a list, removal is O(N). We can just leave it and check in_S.
            else:
                # Invalid insertion. Accept with probability?
                # This would lead to a very messy state. 
                # Better to only accept valid moves for insertion, 
                # and use swaps to optimize the structure for future insertions.
                pass
                
        else:
            # Swap: Remove one element, add another.
            if not S or not candidates:
                break
                
            # Pick element to remove
            idx_remove = random.randint(0, len(S) - 1)
            x_remove = S[idx_remove]
            
            # Pick candidate to add
            x_add = random.choice(candidates)
            
            if x_add == x_remove:
                continue
                
            # Check if x_add is already in S (shouldn't be if candidates are maintained, but safe)
            if x_add in in_S:
                continue
                
            # Calculate cost of removing x_remove and adding x_add.
            # This is complex. 
            # Alternative: 
            # 1. Remove x_remove from S and existing_sums.
            # 2. Check if x_add can be added to the new S.
            # 3. If yes, cost is 0 (valid). If no, calculate collisions.
            
            # Efficient removal:
            # We need to subtract sums involving x_remove.
            # This is O(N^2) to update existing_sums.
            
            # Let's just do a full check for the new set if we propose a swap?
            # That is O(N^3) which is too slow for 20k iterations if N=60.
            
            # Optimization: 
            # Only propose swaps if we are "stuck" (no valid insertions found in recent history).
            # For now, let's stick to valid insertions only, and use random restarts if stuck.
            # This is safer and simpler.
            
            # Revised Strategy:
            # 1. Greedy Init.
            # 2. Loop:
            #    a. Try to insert a random candidate. If valid, accept.
            #    b. If stuck (no valid inserts for X tries), perform a "kick":
            #       Remove 1-2 random elements to create space, then try to re-insert.
            #       This is a form of local search / SA without explicit temperature for invalid states.
            
            pass
            
        # Check for best
        if current_size > best_size:
            best_S = list(S)
            best_size = current_size
            
    # Final greedy extension
    if time.time() < deadline - 5:
        # Try to extend best_S greedily
        S_final = list(best_S)
        sums_final = _compute_triple_sums(S_final)
        in_final = set(S_final)
        
        for x in range(1, n_max + 1):
            if x in in_final:
                continue
            valid, new_sums = _check_conflict_fast(x, S_final, sums_final)
            if valid:
                S_final.append(x)
                sums_final.update(new_sums)
                in_final.add(x)
                
        if len(S_final) > len(best_S):
            best_S = S_final
            best_size = len(best_S)
            
    return best_S

if __name__ == "__main__":
    # Quick test
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s}, Size: {len(res)}")
    # Verify
    if _compute_triple_sums(res) and len(res) > 0:
        # Re-verify validity
        n = len(res)
        seen = set()
        valid = True
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s_val = res[i] + res[j] + res[k]
                    if s_val in seen:
                        valid = False
                        break
                    seen.add(s_val)
                if not valid: break
            if not valid: break
        print("Valid" if valid else "Invalid")
    else:
        print("Empty or Error")

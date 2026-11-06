import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    # State representation
    # S: list of elements
    # sum_counts: dict sum -> count
    # distinct_sums: int
    
    def init_state(S):
        sum_counts = {}
        distinct = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s not in sum_counts:
                    sum_counts[s] = 1
                    distinct += 1
                else:
                    sum_counts[s] += 1
        return sum_counts, distinct

    def get_conflicts(sum_counts):
        # Number of "extra" sums due to duplicates
        # If a sum has count k, it contributes k-1 to conflicts relative to distinct count
        # Total pairs = N(N+1)/2
        # Distinct = D
        # Conflicts = Total - D
        # We want to minimize Conflicts
        return 0 # Placeholder, we calculate delta directly

    def try_move_add(x, S, sum_counts, distinct):
        """
        Attempt to add x to S.
        Returns (new_S, new_sum_counts, new_distinct, delta_conflicts)
        delta_conflicts is the change in (TotalPairs - Distinct)
        Actually, let's just return the new state and let the caller evaluate.
        """
        new_S = S + [x]
        new_sum_counts = sum_counts.copy()
        new_distinct = distinct
        
        # Check x + x
        s = 2 * x
        if s in new_sum_counts:
            new_sum_counts[s] += 1
            # Distinct doesn't change, but total pairs increases by 1
            # So conflicts increase by 1
        else:
            new_sum_counts[s] = 1
            new_distinct += 1
            # Total pairs +1, Distinct +1 -> Conflicts unchanged
            
        # Check x + y for y in S
        for y in S:
            s = x + y
            if s in new_sum_counts:
                new_sum_counts[s] += 1
            else:
                new_sum_counts[s] = 1
                new_distinct += 1
        
        return new_S, new_sum_counts, new_distinct

    def try_move_remove(x, S, sum_counts, distinct):
        new_S = [v for v in S if v != x]
        new_sum_counts = sum_counts.copy()
        new_distinct = distinct
        
        s = 2 * x
        new_sum_counts[s] -= 1
        if new_sum_counts[s] == 0:
            del new_sum_counts[s]
            new_distinct -= 1
        # If count > 0, distinct unchanged. Total pairs -1. Conflicts -1.
        
        for y in new_S:
            s = x + y
            new_sum_counts[s] -= 1
            if new_sum_counts[s] == 0:
                del new_sum_counts[s]
                new_distinct -= 1
        
        return new_S, new_sum_counts, new_distinct

    def try_move_swap(x, y, S, sum_counts, distinct):
        # Remove x, Add y
        # Equivalent to remove then add, but we can optimize
        # For now, just chain them
        S1, sc1, d1 = try_move_remove(x, S, sum_counts, distinct)
        S2, sc2, d2 = try_move_add(y, S1, sc1, d1)
        return S2, sc2, d2

    def objective(S, sum_counts, distinct):
        n = len(S)
        total_pairs = n * (n + 1) // 2
        conflicts = total_pairs - distinct
        # We want to maximize size, minimize conflicts
        # Score = size - lambda * conflicts
        # But we need to balance.
        # If we find a valid set (conflicts=0), we want to maximize size.
        # If we are in a conflict state, we want to reduce conflicts.
        
        # A simple heuristic: 
        # Primary: Maximize size
        # Secondary: Minimize conflicts
        # But if we prioritize size, we might get stuck with many conflicts.
        # Let's use a weighted score.
        # If conflicts > 0, we should strongly penalize.
        
        # Actually, for SA, we can just use:
        # J = - (conflicts * 1000) + len(S)
        # This ensures we prefer valid sets of any size over invalid sets,
        # and among valid sets, we prefer larger ones.
        # Among invalid sets, we prefer fewer conflicts.
        
        return -conflicts * 1000 + n

    # Greedy initialization
    def greedy_init():
        S = list(prescribed)
        S.sort()
        sum_counts, distinct = init_state(S)
        
        # Try to add elements greedily
        # Order: random or ascending?
        # Let's try a few orders
        best_S = S
        best_sc = sum_counts
        best_d = distinct
        
        for order in ['random', 'ascending', 'descending']:
            S_temp = list(prescribed)
            S_temp.sort()
            sc_temp, d_temp = init_state(S_temp)
            
            if order == 'random':
                candidates = list(range(1, max_val + 1))
                random.shuffle(candidates)
            elif order == 'ascending':
                candidates = list(range(1, max_val + 1))
            else:
                candidates = list(range(max_val, 0, -1))
            
            S_set = set(S_temp)
            for x in candidates:
                if x in S_set:
                    continue
                # Check if x can be added without creating conflicts
                # We need to check if 2x and x+y are in sum_counts
                can_add = True
                if 2*x in sc_temp:
                    can_add = False
                else:
                    for y in S_temp:
                        if x+y in sc_temp:
                            can_add = False
                            break
                
                if can_add:
                    # Add
                    S_temp.append(x)
                    S_temp.sort()
                    S_set.add(x)
                    # Update sums
                    s = 2*x
                    sc_temp[s] = 1
                    d_temp += 1
                    for y in S_temp:
                        if y == x: continue
                        s = x + y
                        sc_temp[s] = 1
                        d_temp += 1
            
            if len(S_temp) > len(best_S):
                best_S = S_temp
                best_sc = sc_temp
                best_d = d_temp
                
        return best_S, best_sc, best_d

    # Simulated Annealing
    def simulated_annealing(S, sum_counts, distinct):
        best_S = S
        best_sc = sum_counts
        best_d = distinct
        best_obj = objective(S, sum_counts, distinct)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        # Precompute all possible moves? No, generate on the fly.
        
        while T > min_T and time.time() < deadline - 5:
            # Generate a move
            move_type = random.random()
            S_set = set(S)
            
            if move_type < 0.3:
                # Add
                candidates = [x for x in all_vals if x not in S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_S, new_sc, new_d = try_move_add(x, S, sum_counts, distinct)
            elif move_type < 0.6:
                # Remove
                removable = [x for x in S if x not in prescribed_set]
                if not removable:
                    continue
                x = random.choice(removable)
                new_S, new_sc, new_d = try_move_remove(x, S, sum_counts, distinct)
            else:
                # Swap
                removable = [x for x in S if x not in prescribed_set]
                if not removable:
                    continue
                x = random.choice(removable)
                candidates = [y for y in all_vals if y not in S_set]
                if not candidates:
                    continue
                y = random.choice(candidates)
                new_S, new_sc, new_d = try_move_swap(x, y, S, sum_counts, distinct)
            
            new_obj = objective(new_S, new_sc, new_d)
            delta_obj = new_obj - best_obj # Wait, delta should be vs current state, not best
            # Correction: delta_obj = new_obj - current_obj
            
            # Recalculate current_obj? No, we can store it.
            # Let's store current_obj
            # Actually, objective is cheap.
            
            # Let's re-evaluate:
            # We need current_obj.
            
            # To avoid recalculating, let's just use the delta from the move logic?
            # No, objective is simple.
            
            # Let's refactor to keep current_obj.
            
            # For now, just calculate delta vs current state.
            # I'll implement a proper loop below.
            
            pass # Placeholder

    # Proper SA Loop
    def run_sa(S, sum_counts, distinct):
        current_S = S
        current_sc = sum_counts
        current_d = distinct
        current_obj = objective(current_S, current_sc, current_d)
        
        best_S = current_S.copy()
        best_sc = current_sc.copy()
        best_d = current_d
        best_obj = current_obj
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        
        S_set = set(current_S)
        
        while T > min_T and time.time() < deadline - 10:
            # Generate move
            r = random.random()
            if r < 0.3:
                # Add
                candidates = [x for x in all_vals if x not in S_set]
                if not candidates:
                    T *= cooling
                    continue
                x = random.choice(candidates)
                new_S, new_sc, new_d = try_move_add(x, current_S, current_sc, current_d)
                new_S_set = S_set.union([x])
            elif r < 0.6:
                # Remove
                removable = [x for x in current_S if x not in prescribed_set]
                if not removable:
                    T *= cooling
                    continue
                x = random.choice(removable)
                new_S, new_sc, new_d = try_move_remove(x, current_S, current_sc, current_d)
                new_S_set = S_set.difference([x])
            else:
                # Swap
                removable = [x for x in current_S if x not in prescribed_set]
                if not removable:
                    T *= cooling
                    continue
                x = random.choice(removable)
                candidates = [y for y in all_vals if y not in S_set]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                new_S, new_sc, new_d = try_move_swap(x, y, current_S, current_sc, current_d)
                new_S_set = S_set.difference([x]).union([y])
            
            new_obj = objective(new_S, new_sc, new_d)
            delta_obj = new_obj - current_obj
            
            if delta_obj >= 0 or random.random() < math.exp(delta_obj / T):
                # Accept
                current_S = new_S
                current_sc = new_sc
                current_d = new_d
                current_obj = new_obj
                S_set = new_S_set
                
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_S = new_S.copy()
                    best_sc = new_sc.copy()
                    best_d = new_d
                    
            T *= cooling
            
        return best_S, best_sc, best_d

    # Main
    best_S, best_sc, best_d = greedy_init()
    best_size = len(best_S)
    
    # Run SA multiple times with different restarts or just one long run?
    # One long run with high initial T might be better.
    # But greedy init is already good.
    
    # Let's run SA on the greedy result
    final_S, _, _ = run_sa(best_S, best_sc, best_d)
    
    if len(final_S) > best_size:
        best_S = final_S
        best_size = len(best_S)
        
    # If we didn't reach 20, try more aggressive restarts
    # But SA should be robust.
    
    return best_S

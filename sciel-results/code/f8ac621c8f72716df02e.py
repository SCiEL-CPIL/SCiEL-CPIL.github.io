"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Calculate LCM of 1..30
    L = reduce(lcm, range(1, 31))
    # Weights w[i] corresponds to coefficient c[i] (0-indexed) for term 1/(i+1)
    # Sum = sum(c[i] * L/(i+1)) / L
    # We minimize abs(sum(c[i] * L/(i+1)))
    weights = [L // (i + 1) for i in range(30)]
    
    def compute_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return s

    def local_search(coeffs, current_sum):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    break
        return coeffs, current_sum

    best_coeffs = None
    best_sum = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_coeffs, best_sum
        if int_sum == 0:
            return
        if best_sum is None or abs(int_sum) < abs(best_sum):
            best_sum = int_sum
            best_coeffs = list(coeffs)

    def tabu_search(initial_coeffs, initial_sum, tabu_tenure=10, max_iter=5000):
        coeffs = list(initial_coeffs)
        current_sum = initial_sum
        current_abs = abs(current_sum)
        
        # Tabu list: stores (index, old_value) that are tabu until iteration 'until'
        # We use a dictionary: key = (index, old_value), value = until_iteration
        # Or simpler: list of (index, old_value, until_iter)
        # For efficiency, let's use a list of moves and check.
        # Actually, a move is defined by (index, new_value). The reverse move is (index, old_value).
        # Let's store (index, old_value) in a set/dict with expiration.
        
        tabu_dict = {}
        best_local_sum = current_sum
        best_local_coeffs = list(coeffs)
        
        no_improve_count = 0
        max_no_improve = 1000
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Clean up expired tabu moves
            if it % 10 == 0:
                keys_to_remove = [k for k, v in tabu_dict.items() if v < it]
                for k in keys_to_remove:
                    del tabu_dict[k]
                    
            best_move = None
            best_move_score = float('inf')
            best_move_is_tabu = False
            best_move_tabu_until = 0
            
            for i in range(30):
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    
                    # Check if this move is tabu
                    # A move (i, old_val -> new_val) is tabu if the reverse move (i, new_val -> old_val) was made recently?
                    # Standard Tabu: The move just made is tabu.
                    # Move: change c[i] from old_val to new_val.
                    # The reverse move would be changing c[i] from new_val to old_val.
                    # We mark (i, new_val) as the state we are leaving? No.
                    # We mark the move itself.
                    # Let's define a move by (index, new_value).
                    # If we just changed c[i] to new_val, we don't want to immediately change it back to old_val.
                    # So we add (i, old_val) to the tabu list? No, we want to forbid the move that restores the old state.
                    # The move we are considering is (i, new_val).
                    # The move that is tabu is the one that was just performed.
                    # If we perform move M = (i, new_val), then the reverse move (i, old_val) becomes tabu?
                    # Actually, if we are at state S and move to S' by changing c[i] to v, we forbid changing c[i] back to old_v for T steps.
                    # So if we are considering changing c[i] from current_val to new_val,
                    # we check if (i, current_val) is in the tabu list?
                    # No, if we just moved from A to B, we forbid B to A.
                    # So if we are at B and want to go to A, we check if (i, A) is tabu?
                    # Let's say we store (index, value_to_restore) in tabu list.
                    # If we change c[i] from 1 to 0, we add (i, 1) to tabu list.
                    # Now, if we consider changing c[i] from 0 to 1, we see (i, 1) is in tabu list, so we forbid it.
                    
                    is_tabu = (i, old_val) in tabu_dict and tabu_dict[(i, old_val)] > it
                    # Wait, if we just moved to 'new_val', the 'old_val' is the one we came from.
                    # So we forbid moving back to 'old_val'.
                    # So the key in tabu dict should be (i, old_val) representing "do not set c[i] to old_val".
                    
                    if is_tabu:
                        # Aspiration criterion: allow if it improves global best
                        pass # We'll handle aspiration below or just skip for simplicity first
                    
                    new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                    if new_sum == 0:
                        continue
                    
                    new_abs = abs(new_sum)
                    
                    # Aspiration: if this move leads to a better solution than best_local_sum, allow it even if tabu
                    if new_abs < abs(best_local_sum):
                        is_tabu = False
                    
                    if not is_tabu:
                        if new_abs < best_move_score:
                            best_move_score = new_abs
                            best_move = (i, new_val)
                            best_move_is_tabu = False
                    else:
                        # Track best tabu move in case all moves are tabu
                        if new_abs < best_move_score:
                            best_move_score = new_abs
                            best_move = (i, new_val)
                            best_move_is_tabu = True
                            best_move_tabu_until = tabu_dict.get((i, old_val), it + tabu_tenure)

            if best_move is None:
                break
            
            i, new_val = best_move
            old_val = coeffs[i]
            
            # Perform move
            coeffs[i] = new_val
            current_sum = current_sum - old_val * weights[i] + new_val * weights[i]
            current_abs = abs(current_sum)
            
            # Add to tabu list
            # We just moved TO new_val FROM old_val.
            # We want to forbid moving back TO old_val.
            # So we mark (i, old_val) as tabu.
            tabu_dict[(i, old_val)] = it + tabu_tenure
            
            # Update best local
            if current_abs < abs(best_local_sum):
                best_local_sum = current_sum
                best_local_coeffs = list(coeffs)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break

        return best_local_coeffs, best_local_sum

    # Initialization strategies
    def random_init(num_nonzero):
        coeffs = [0] * 30
        indices = random.sample(range(30), num_nonzero)
        for i in indices:
            coeffs[i] = random.choice([-1, 1])
        s = compute_sum(coeffs)
        if s == 0:
            # Perturb
            i = random.randint(0, 29)
            coeffs[i] = random.choice([-1, 1])
            s = compute_sum(coeffs)
            if s == 0:
                coeffs[0] = 1
                s = weights[0]
        return coeffs, s

    def greedy_init():
        coeffs = [0] * 30
        current_sum = 0
        indices = sorted(range(30), key=lambda i: weights[i], reverse=True)
        for i in indices:
            best_val = 0
            best_abs = abs(current_sum) if current_sum != 0 else float('inf')
            for val in (-1, 0, 1):
                new_sum = current_sum + val * weights[i]
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_val = val
            coeffs[i] = best_val
            current_sum += best_val * weights[i]
        if current_sum == 0:
            coeffs[0] = 1
            current_sum = weights[0]
        return coeffs, current_sum

    random.seed(42)
    
    # 1. Greedy Init + Local Search
    coeffs, s = greedy_init()
    update_best(coeffs, s)
    coeffs, s = local_search(coeffs, s)
    update_best(coeffs, s)
    
    # 2. Multiple Tabu Search Restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Vary initialization
        if r % 3 == 0:
            coeffs, s = random_init(random.randint(5, 15))
        elif r % 3 == 1:
            coeffs, s = random_init(random.randint(10, 20))
        else:
            # Start from current best with perturbation
            if best_coeffs:
                coeffs = list(best_coeffs)
                # Flip 2-3 random bits
                for _ in range(3):
                    i = random.randint(0, 29)
                    coeffs[i] = random.choice([-1, 0, 1])
                s = compute_sum(coeffs)
                if s == 0:
                    coeffs[0] = 1
                    s = weights[0]
            else:
                coeffs, s = random_init(10)
                
        update_best(coeffs, s)
        
        # Run Tabu Search
        # Adjust tenure and iterations based on time left
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Heuristic for iteration count: more time -> more iterations
        # But we also have a hard limit per restart to allow many restarts
        max_iter = 2000
        tabu_tenure = 15
        
        coeffs, s = tabu_search(coeffs, s, tabu_tenure, max_iter)
        update_best(coeffs, s)
        
        # Final local search
        coeffs, s = local_search(coeffs, s)
        update_best(coeffs, s)

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END

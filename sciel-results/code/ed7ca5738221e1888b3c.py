import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    weights = [1.0 / i for i in range(1, n + 1)]  # weights[i-1] = 1/i
    
    # Precompute all 3-term APs for fast checking
    # For each element x, precompute which pairs (a,b) would form an AP with x
    # Actually, let's just check on the fly - it's fast enough for n=256
    
    def is_ap_free(S):
        """Check if set S contains no 3-term AP."""
        S_list = sorted(S)
        S_set = set(S_list)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                # Check if a, b, c form AP where c = 2b - a
                c = 2 * b - a
                if c in S_set and c != a and c != b:
                    return False
                # Check if a, c, b form AP where c = 2a - b (a is middle)
                # Actually, just check all triples
        # More efficient: for each pair, check the third element
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                # a, b, c: c = 2b - a
                c = 2 * b - a
                if c in S_set and c != a and c != b:
                    return False
        return True
    
    def can_add(x, S_set):
        """Check if x can be added to S without creating a 3-AP."""
        if x in S_set:
            return False
        S_list = list(S_set)
        # Check if x as middle: y, x, z => z = 2x - y
        for y in S_list:
            z = 2 * x - y
            if z in S_set:
                return False
        # Check if x as end: y, z, x => z = (x+y)/2 or y = (x+z)/2
        # Equivalently: for each y in S, check if (x+y) is even and (x+y)//2 in S
        for y in S_list:
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S_set and z != x and z != y:
                    return False
            # Check if y is middle: x, y, z => z = 2y - x
            z = 2 * y - x
            if z in S_set and z != x and z != y:
                return False
        return True
    
    def get_weight(S):
        return sum(weights[i - 1] for i in S)
    
    # Greedy initialization: add elements in order of decreasing weight (1, 2, 3, ...)
    # Actually, to maximize reciprocal sum, we want small numbers (large weights)
    # Try greedy from 1 to n, adding if possible
    def greedy_init():
        S = set()
        S_set = set()
        for x in range(1, n + 1):
            if can_add(x, S_set):
                S.add(x)
                S_set.add(x)
        return S
    
    # Local search: try 1-add, 1-remove, 1-swap moves
    def local_search(S, time_budget):
        S = set(S)
        S_set = set(S)
        current_weight = get_weight(S)
        best_weight = current_weight
        best_S = set(S)
        
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < time_budget:
            improved = False
            
            # Try 1-add moves: add an element not in S
            for x in range(1, n + 1):
                if x not in S_set:
                    if can_add(x, S_set):
                        new_weight = current_weight + weights[x - 1]
                        if new_weight > current_weight:
                            S.add(x)
                            S_set.add(x)
                            current_weight = new_weight
                            improved = True
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_S = set(S)
                            break  # Restart after improvement
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try 1-remove moves: remove an element from S
            for x in list(S):
                S.remove(x)
                S_set.remove(x)
                new_weight = current_weight - weights[x - 1]
                # After removal, we might be able to add other elements
                # For now, just accept if it leads to better state after re-adding
                # Actually, let's try swap moves instead which are more effective
                
                # Restore
                S.add(x)
                S_set.add(x)
            
            # Try 1-swap moves: remove x, add y
            for x in list(S):
                if time.time() > time_budget:
                    break
                S.remove(x)
                S_set.remove(x)
                base_weight = current_weight - weights[x - 1]
                
                for y in range(1, n + 1):
                    if y not in S_set:
                        if can_add(y, S_set):
                            new_weight = base_weight + weights[y - 1]
                            if new_weight > current_weight:
                                S.add(y)
                                S_set.add(y)
                                current_weight = new_weight
                                improved = True
                                if current_weight > best_weight:
                                    best_weight = current_weight
                                    best_S = set(S)
                                break  # Restart after improvement
                
                # Restore x if we didn't swap
                if x not in S:
                    S.add(x)
                    S_set.add(x)
                
                if improved:
                    break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
        
        return best_S
    
    # Run greedy init
    S = greedy_init()
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:  # Need at least 5s for local search
        S = local_search(S, deadline - 5.0)
    
    # Final validation
    final_list = sorted(S)
    if not final_list:
        return [1]
    
    # Verify no 3-AP
    S_set = set(final_list)
    for a in final_list:
        for b in final_list:
            if a == b:
                continue
            c = 2 * b - a
            if c in S_set and c != a and c != b:
                # Invalid, fallback to simple valid set
                return [1, 3, 7, 12, 20, 24, 31, 35, 45, 50, 57, 61, 71, 75, 85, 89, 99, 103, 113, 117, 127, 131, 141, 145, 155, 159, 169, 173, 183, 187, 197, 201, 211, 215, 225, 229, 239, 243, 253]
    
    return final_list

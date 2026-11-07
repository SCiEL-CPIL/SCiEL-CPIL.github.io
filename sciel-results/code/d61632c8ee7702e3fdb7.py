import time
import random
import math

P = {'n': 256}

def construct():
    start_time = time.time()
    limit = P['n']
    deadline = start_time + 55  # Safety margin
    
    # Range of numbers
    numbers = list(range(2, limit + 1))
    N = len(numbers)
    
    # Precompute prime factors
    # factors[i] is a dictionary {prime: exponent}
    factors = [{} for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        temp = i
        for p in range(2, int(math.sqrt(temp)) + 1):
            while temp % p == 0:
                factors[i][p] = factors[i].get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[i][temp] = factors[i].get(temp, 0) + 1

    # Helper to check if a | bc
    def divides_product(a, b, c):
        # Check if a divides b*c
        # a | bc iff for all p, v_p(a) <= v_p(b) + v_p(c)
        fa = factors[a]
        fb = factors[b]
        fc = factors[c]
        
        # Get all primes involved in a
        for p, exp_a in fa.items():
            exp_b = fb.get(p, 0)
            exp_c = fc.get(p, 0)
            if exp_a > exp_b + exp_c:
                return False
        return True

    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    # Check a | bc
                    if divides_product(a, b, c):
                        return False
                    # Check b | ac
                    if divides_product(b, a, c):
                        return False
                    # Check c | ab
                    if divides_product(c, a, b):
                        return False
        return True

    def get_conflicts(x, S):
        """
        Returns a list of pairs (b, c) in S such that adding x would create a conflict.
        Conflict if:
        x | bc
        b | xc
        c | xb
        """
        conflicts = []
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(i + 1, n):
                c = S_list[j]
                if divides_product(x, b, c):
                    conflicts.append((b, c))
                elif divides_product(b, x, c):
                    conflicts.append((b, c))
                elif divides_product(c, x, b):
                    conflicts.append((b, c))
        return conflicts

    def local_search(initial_S):
        S = set(initial_S)
        # Ensure initial S is valid
        if not is_valid_set(S):
             # Fix by removing elements causing conflicts
             S_list = list(S)
             for i in range(len(S_list)):
                 if time.time() > deadline: break
                 a = S_list[i]
                 S.remove(a)
                 if is_valid_set(S):
                     S.add(a)
                     # If adding back failed, leave it out? 
                     # Simpler: just return empty or minimal if initial is bad.
                     # But our initial heuristics should be valid.
             # If still invalid, return empty
             if not is_valid_set(S):
                 return set()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            candidates = [x for x in numbers if x not in S]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if not get_conflicts(x, S):
                    S.add(x)
                    improved = True
                    added = True
                    break # Restart loop to re-evaluate other candidates
            
            if added:
                continue
                
            # Phase 2: Try to swap
            # For each y in S, remove it, then try to add any x
            # This is O(|S| * |Candidates| * Cost)
            # To speed up, only try a few swaps per iteration
            
            S_list = list(S)
            random.shuffle(S_list)
            
            swapped = False
            for y in S_list:
                if time.time() > deadline:
                    break
                S.remove(y)
                
                # Try to add any candidate
                candidates = [x for x in numbers if x not in S]
                random.shuffle(candidates)
                
                best_x = None
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if not get_conflicts(x, S):
                        best_x = x
                        break
                
                if best_x is not None:
                    S.add(best_x)
                    # We swapped y for best_x. Size is same.
                    # But this might unlock more additions.
                    improved = True
                    swapped = True
                    break
                else:
                    # Put y back
                    S.add(y)
            
            if not swapped:
                # Try to remove an element to see if we can add TWO?
                # This is deeper search.
                # For now, if no single swap works, we are stuck in a local optimum.
                # Random restart might be better.
                pass
                
        return S

    # Heuristic Initialization
    # 1. All primes > 128
    # 2. All composites > 128
    # 3. Greedily add others
    
    def generate_initial_set():
        S = set()
        # Add large numbers
        for i in range(129, limit + 1):
            S.add(i)
        
        # Greedily add smaller numbers
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in S:
                continue
            if not get_conflicts(x, S):
                S.add(x)
        
        return S

    best_S = set()
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        init_S = generate_initial_set()
        curr_S = local_search(init_S)
        
        if len(curr_S) > len(best_S):
            best_S = curr_S
            
        # If we found a very large set, we might stop early?
        # Max possible is 255.
        # Current best known for similar problems is around 100-150?
        # Let's just keep running.
        
        # Optional: If we are making no progress, break?
        # The local search handles stagnation by swapping.
        
    # Final verification
    if not is_valid_set(best_S):
        # Fallback to a simple valid set
        best_S = set(range(129, limit + 1))
        
    return list(best_S)

# EVOLVE-BLOCK-END

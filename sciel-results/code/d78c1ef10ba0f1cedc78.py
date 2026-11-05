import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 4096
    max_sum = 3 * n
    
    # used_sums[i] is 1 if sum i has been used
    used_sums = bytearray(max_sum + 1)
    
    S = []
    
    # Greedy: try to add smallest possible x
    # We can also try randomized restarts if we have time
    best_S = []
    
    def try_greedy(order):
        local_used = bytearray(max_sum + 1)
        local_S = []
        
        for x in order:
            if time.time() > deadline:
                return local_S
            
            # Check if x can be added
            # New sums:
            # 1. x+x+x
            # 2. x+x+y for y in local_S
            # 3. x+y+z for y,z in local_S, y<=z
            
            new_sums = set()
            
            # 1
            s = 3*x
            if local_used[s]:
                return local_S
            new_sums.add(s)
            
            # 2
            for y in local_S:
                s = 2*x + y
                if local_used[s] or s in new_sums:
                    return local_S
                new_sums.add(s)
                
            # 3
            # We need to be careful with y<=z to avoid duplicates in the set of triples, 
            # but since we are adding to a set of sums, duplicates in generation 
            # just mean we check the same sum twice. 
            # However, if y=z, it's one triple. If y!=z, it's one triple (unordered).
            # The condition is on unordered triples. 
            # So (x,y,z) is the same as (x,z,y).
            # We just need to ensure the sum x+y+z is unique among all triples.
            
            for i in range(len(local_S)):
                y = local_S[i]
                for j in range(i, len(local_S)):
                    z = local_S[j]
                    s = x + y + z
                    if local_used[s] or s in new_sums:
                        return local_S
                    new_sums.add(s)
                    
            # If we reach here, x is valid
            local_S.append(x)
            for s in new_sums:
                local_used[s] = 1
                
        return local_S

    # Standard greedy: 1, 2, ..., 4096
    S1 = try_greedy(range(1, n+1))
    best_S = S1
    
    # Randomized greedy: shuffle the order
    # We can try a few random permutations
    import random
    random.seed(42)
    
    # To save time, we might not try full permutations, but maybe random subsets or steps
    # Given the deadline, let's try a few random full scans if time allows
    # But 4096 steps * O(k^2) is fast.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        order = list(range(1, n+1))
        random.shuffle(order)
        S_rand = try_greedy(order)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            attempts += 1
            # If we found a significantly better one, we might stop or continue
            # Let's continue to see if we can do better
        else:
            attempts += 1
            if attempts > 50: # Limit random attempts to avoid wasting time if greedy is already good
                break
                
    return best_S

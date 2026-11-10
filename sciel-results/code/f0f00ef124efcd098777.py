import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    def try_greedy(order):
        """Try greedy addition in given order."""
        S = [set() for _ in range(61)]
        S[0].add(0)
        chosen = []
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Compute new subset sums if we add x
            new_S = [set(s) for s in S]
            for k in range(len(chosen) + 1, 0, -1):
                new_sums = set(s + x for s in S[k-1])
                new_S[k] = new_S[k] | new_sums
            
            # Check disjointness
            valid = True
            max_k = len(chosen) + 1
            for k in range(0, max_k + 1):
                if not new_S[k]:
                    continue
                for j in range(k + 1, max_k + 1):
                    if not new_S[j]:
                        continue
                    if new_S[k] & new_S[j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S = new_S
                chosen.append(x)
        
        return chosen
    
    # Try ascending order first
    result_asc = try_greedy(range(1, 257))
    
    # Try descending order if time permits
    if time.time() < deadline - 2:
        result_desc = try_greedy(range(256, 0, -1))
        if len(result_desc) > len(result_asc):
            result_asc = result_desc
    
    return result_asc

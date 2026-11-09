import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    N = 512
    elements = list(range(1, N + 1))
    
    def is_4ap_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if b <= a:
                    continue
                d = b - a
                if d == 0:
                    continue
                # Check a, b, b+d, b+2d
                # Terms: a, a+d, a+2d, a+3d
                # We have a and b=a+d.
                # Need a+2d and a+3d in S.
                if (a + 2*d) in S_set and (a + 3*d) in S_set:
                    return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            # Check if adding x creates a 4-AP
            # x can be term 1, 2, 3, or 4.
            # 1: x, x+d, x+2d, x+3d -> need 3 in S
            # 2: x-d, x, x+d, x+2d -> need 2 in S
            # 3: x-2d, x-d, x, x+d -> need 2 in S
            # 4: x-3d, x-2d, x-d, x -> need 3 in S
            
            valid = True
            # x is 1st term: x, x+d, x+2d, x+3d
            # d >= 1
            # x+3d <= N
            max_d1 = (N - x) // 3
            for d in range(1, max_d1 + 1):
                if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                    valid = False
                    break
            if not valid:
                continue
                
            # x is 2nd term: x-d, x, x+d, x+2d
            # x-d >= 1 => d <= x-1
            # x+2d <= N => d <= (N-x)//2
            max_d2 = min(x - 1, (N - x) // 2)
            for d in range(1, max_d2 + 1):
                if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                    valid = False
                    break
            if not valid:
                continue
                
            # x is 3rd term: x-2d, x-d, x, x+d
            # x-2d >= 1 => d <= (x-1)//2
            # x+d <= N => d <= N-x
            max_d3 = min((x - 1) // 2, N - x)
            for d in range(1, max_d3 + 1):
                if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                    valid = False
                    break
            if not valid:
                continue
                
            # x is 4th term: x-3d, x-2d, x-d, x
            # x-3d >= 1 => d <= (x-1)//3
            max_d4 = (x - 1) // 3
            for d in range(1, max_d4 + 1):
                if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                    valid = False
                    break
            
            if valid:
                S.add(x)
                
        return S

    best_set = set()
    best_size = 0
    
    # 1. Deterministic Greedy
    S1 = greedy_build(elements)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
        
    S2 = greedy_build(elements[::-1])
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
        
    # 2. Random Greedy
    random.seed(42)
    num_randoms = 200
    for i in range(num_randoms):
        if time.time() > deadline:
            break
        random.shuffle(elements)
        Sr = greedy_build(elements)
        if len(Sr) > best_size:
            best_set = Sr
            best_size = len(Sr)
            
    # 3. Local Search / Improvement
    # Try to add any remaining elements
    def try_add(S, x):
        # Re-check validity for x given S
        valid = True
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return False
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return False
        return True

    S_best = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any excluded element
        for x in range(1, N + 1):
            if x not in S_best:
                if try_add(S_best, x):
                    S_best.add(x)
                    improved = True
                    # Restart loop to try more adds
                    break
        
        if not improved:
            # Try swaps
            # To avoid O(N^2) heavy checks, we can try random swaps
            # If we find a swap that keeps validity, we accept it.
            # Then we try to add again.
            # We need to be careful not to oscillate.
            # Simple heuristic: Accept swap if it allows an add in the next step.
            
            found_swap = False
            # Sample a few random swaps
            for _ in range(100):
                if time.time() > deadline:
                    break
                if not S_best:
                    break
                u = random.choice(list(S_best))
                # Pick a random v not in S_best
                # Efficient way: pick random from 1..N, if in S, retry or pick next
                v = random.randint(1, N)
                if v in S_best:
                    continue
                if u == v:
                    continue
                
                # Test S' = S_best - {u} + {v}
                S_temp = S_best.copy()
                S_temp.remove(u)
                S_temp.add(v)
                
                # Check if S_temp is 4-AP free
                # This is O(|S|^2) which is ~100^2 = 10k ops. 100 swaps = 1M ops. OK.
                if is_4ap_free(S_temp):
                    # Try to add something to S_temp
                    added = False
                    for x in range(1, N + 1):
                        if x not in S_temp:
                            if try_add(S_temp, x):
                                S_temp.add(x)
                                added = True
                                break
                    
                    if added:
                        S_best = S_temp
                        improved = True
                        found_swap = True
                        break
                        
            if found_swap:
                continue
            else:
                break # No improvement found in this iteration of swaps
            
    return sorted(S_best)

if __name__ == "__main__":
    print(construct())

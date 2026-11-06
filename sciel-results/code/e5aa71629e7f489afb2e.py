import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 512
    TARGET_SIZE = 48
    
    def get_violation_counts(S):
        n = len(S)
        if n < 3:
            return {x: 0 for x in S}
        
        violation_counts = {x: 0 for x in S}
        
        for x in S:
            others = [v for v in S if v != x]
            m = len(others)
            
            dp = [0] * (m + 1)
            dp[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            
            for j in range(2, m + 1):
                target_sum = j * x
                if (dp[j] >> target_sum) & 1:
                    violation_counts[x] += 1
                    
        return violation_counts

    def is_valid(S):
        if len(S) < 3:
            return True
        counts = get_violation_counts(S)
        return all(c == 0 for c in counts.values())

    def _is_valid_insertion(S, x):
        n = len(S)
        
        # Check 1: x is mean of subset of S
        if n >= 2:
            dp_S = [0] * (n + 1)
            dp_S[0] = 1
            for e in S:
                for k in range(n, 0, -1):
                    dp_S[k] |= (dp_S[k-1] << e)
            
            for k in range(2, n + 1):
                target = k * x
                if (dp_S[k] >> target) & 1:
                    return False
        
        # Check 2: a is mean of subset of (S U {x}) \ {a} involving x
        for a in S:
            others = [v for v in S if v != a]
            m = len(others)
            if m < 1:
                continue
            
            dp_others = [0] * (m + 1)
            dp_others[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp_others[k] |= (dp_others[k-1] << e)
            
            for j in range(1, m + 1):
                target_sum = j * a - x
                if target_sum < 0:
                    continue
                if (dp_others[j] >> target_sum) & 1:
                    return False
        
        return True

    best_set = []
    best_len = 0
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        S = random.sample(range(1, 513), TARGET_SIZE)
        
        # Phase 1: Remove worst until valid
        # Limit time for this phase
        phase1_deadline = time.time() + 2.0
        
        while time.time() < phase1_deadline:
            if is_valid(S):
                break
            counts = get_violation_counts(S)
            worst_elem = max(counts, key=counts.get)
            if counts[worst_elem] == 0:
                break
            S.remove(worst_elem)
            
        if not is_valid(S):
            continue
            
        # Phase 2: Add back to reach TARGET_SIZE
        phase2_deadline = time.time() + 3.0
        
        while len(S) < TARGET_SIZE and time.time() < phase2_deadline:
            added = False
            # Try adding
            for _ in range(50):
                x = random.randint(1, 512)
                if x in S:
                    continue
                if _is_valid_insertion(S, x):
                    S.append(x)
                    added = True
                    break
            
            if not added:
                # Try swap
                swapped = False
                for _ in range(50):
                    a = random.choice(S)
                    x = random.randint(1, 512)
                    if x in S:
                        continue
                    others = [v for v in S if v != a]
                    if _is_valid_insertion(others, x):
                        S.remove(a)
                        S.append(x)
                        swapped = True
                        break
                
                if not swapped:
                    break
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
            
        if best_len == TARGET_SIZE:
            break
            
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")

import random
import time

def construct():
    n = 12
    # Start with powers of 2
    S = [1 << i for i in range(n)]
    best_S = S[:]
    best_max = max(S)
    
    def is_valid(s):
        if len(set(s)) != n:
            return False
        sums = [0]
        for x in s:
            new_sums = [sum_val + x for sum_val in sums]
            sums += new_sums
            # Check for duplicates early
            if len(sums) != len(set(sums)):
                return False
        return True
        
    start_time = time.time()
    deadline = start_time + 50
    
    current_S = S[:]
    current_max = max(S)
    T = 2.0
    cooling = 0.995
    
    while time.time() < deadline:
        move_type = random.random()
        new_S = current_S[:]
        
        if move_type < 0.7:
            # Change one element
            i = random.randint(0, n-1)
            # Bias towards larger indices for reduction
            if i > n//2:
                delta = random.randint(-20, 5)
            else:
                delta = random.randint(-5, 5)
            
            new_val = current_S[i] + delta
            if new_val <= 0:
                continue
            if new_val in current_S and current_S[i] != new_val:
                continue
                
            new_S[i] = new_val
        else:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            new_S[i], new_S[j] = new_S[j], new_S[i]
            
        if is_valid(new_S):
            new_max = max(new_S)
            delta_max = new_max - current_max
            
            if delta_max < 0:
                # Improvement
                current_S = new_S
                current_max = new_max
                if current_max < best_max:
                    best_S = current_S[:]
                    best_max = current_max
            else:
                # Worse or equal
                # Accept with probability based on temperature
                if delta_max == 0:
                    current_S = new_S
                else:
                    prob = 1.0 / (1.0 + delta_max / T)
                    if random.random() < prob:
                        current_S = new_S
                        current_max = new_max
        else:
            # Invalid
            # Reject
            pass
            
        T *= cooling
        if T < 0.05:
            T = 0.05
            
    # Final polish: try to greedily reduce elements
    # We do this on the best solution found
    polish_S = best_S[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(polish_S)):
            # Try to decrease polish_S[i] by small amounts
            for delta in range(1, 15):
                new_val = polish_S[i] - delta
                if new_val <= 0:
                    break
                # Check distinctness
                if new_val in polish_S and polish_S[i] != new_val:
                    continue
                new_S = polish_S[:]
                new_S[i] = new_val
                if is_valid(new_S):
                    polish_S = new_S
                    improved = True
                    break
    
    if max(polish_S) < best_max:
        best_S = polish_S
        best_max = max(polish_S)
        
    return best_S

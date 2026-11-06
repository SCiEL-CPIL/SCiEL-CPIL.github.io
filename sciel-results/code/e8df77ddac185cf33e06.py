import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 512
    K = 32
    
    # Precompute slices for each d to speed up scoring
    # For a given d, we check indices i such that i, i+d, i+2d are valid
    # i ranges from 0 to N - 2d - 1
    # Slices: B[0 : N-2d], B[d : N-d], B[2d : N]
    
    def score_set(B):
        """
        B: numpy bool array of length N
        Returns: number of distinct d > 0 such that a 3-AP with difference d exists
        """
        count = 0
        for d in range(1, N // 2 + 1):
            # Check if there is any alignment where B[i], B[i+d], B[i+2d] are all True
            # Using vectorized AND
            # Slices must be of same length
            # Length of B[:-2*d] is N - 2d
            # Length of B[d:-d] is N - 2d
            # Length of B[2*d:] is N - 2d
            if np.any(B[:-2*d] & B[d:-d] & B[2*d:]):
                count += 1
        return count

    def greedy_init(seed=42):
        """
        Greedily build a set of K elements to maximize score.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        S = set()
        B = np.zeros(N, dtype=bool)
        
        # Start with a random element to break symmetry
        S.add(random.randint(1, N))
        B[S.pop() - 1] = True # Wait, S.pop() modifies S. Let's do this properly.
        
        # Reset
        S = set()
        B = np.zeros(N, dtype=bool)
        
        # Add first element randomly
        first = random.randint(1, N)
        S.add(first)
        B[first - 1] = True
        
        for _ in range(K - 1):
            best_val = -1
            best_score = -1
            
            # Try all candidates not in S
            # Optimization: We can just iterate 1..N and skip if in S
            # Since N=512, this is fast.
            for val in range(1, N + 1):
                if val in S:
                    continue
                
                # Temporarily add val
                B[val - 1] = True
                sc = score_set(B)
                B[val - 1] = False
                
                if sc > best_score:
                    best_score = sc
                    best_val = val
            
            if best_val != -1:
                S.add(best_val)
                B[best_val - 1] = True
            else:
                # Fallback (should not happen)
                candidates = [v for v in range(1, N + 1) if v not in S]
                if candidates:
                    val = random.choice(candidates)
                    S.add(val)
                    B[val - 1] = True
        
        return list(S)

    # Initial best set
    best_set = greedy_init(seed=42)
    best_B = np.zeros(N, dtype=bool)
    for x in best_set:
        best_B[x - 1] = True
    best_score = score_set(best_B)
    
    # Simulated Annealing
    current_set = list(best_set)
    current_B = best_B.copy()
    current_score = best_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9999
    
    # Precompute indices for faster swap? 
    # We maintain a list of elements in S and a boolean array.
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor: swap one element in S with one not in S
        idx = random.randint(0, K - 1)
        old_val = current_set[idx]
        
        # Pick a random new_val not in current_set
        # Efficient way: generate random until unique? Or maintain a list of available?
        # Maintaining a list of available is O(N) to update.
        # Generating random is O(1) expected.
        new_val = random.randint(1, N)
        while new_val in current_set:
            new_val = random.randint(1, N)
        
        # Apply swap to B
        current_B[old_val - 1] = False
        current_B[new_val - 1] = True
        
        new_score = score_set(current_B)
        delta = new_score - current_score
        
        if delta > 0 or (delta >= -5 and random.random() < np.exp(delta / T)):
            # Accept
            current_set[idx] = new_val
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
                best_B = current_B.copy()
        else:
            # Reject: revert B
            current_B[new_val - 1] = False
            current_B[old_val - 1] = True
        
        T *= cooling
        
        # Occasionally restart from best if stuck for too long?
        # Simple restart logic: if T is very low and we haven't improved in a while, reheat
        # But we don't track "improvement counter" easily here.
        # Let's add a simple reheat if T < 1.0 and random chance
        if T < 1.0 and random.random() < 0.001:
            T = 10.0
            # Reset current to best
            current_set = best_set.copy()
            current_B = best_B.copy()
            current_score = best_score

    return best_set

# Quick self-test
if __name__ == "__main__":
    import time
    s = time.time()
    res = construct()
    print(f"Time: {time.time() - s:.2f}s")
    print(f"Result: {res}")
    print(f"Len: {len(res)}")
    print(f"Distinct: {len(set(res)) == 32}")
    print(f"Range: {min(res) >= 1 and max(res) <= 512}")
    
    # Calculate score for verification
    B = np.zeros(512, dtype=bool)
    for x in res:
        B[x-1] = True
    sc = 0
    for d in range(1, 256):
        if np.any(B[:-2*d] & B[d:-d] & B[2*d:]):
            sc += 1
    print(f"Score: {sc}")

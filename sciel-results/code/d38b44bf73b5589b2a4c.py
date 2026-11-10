import time
import random

def construct():
    N = 512
    deadline = time.time() + 55.0 # 55s to be safe within 60s limit
    
    # Use a bytearray for fast membership testing
    # S[x] == 1 if x is in the set, 0 otherwise
    # We use 1-based indexing for elements 1..N
    
    def can_add(x, S):
        # Check if adding x creates a 4-AP
        
        # x is 4th term: x-3d, x-2d, x-d in S
        # d such that x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if S[x - d] and S[x - 2*d] and S[x - 3*d]:
                return False
        
        # x is 3rd term: x-2d, x-d, x+d in S
        # d such that x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
        # and x+d <= N => d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x - 2*d] and S[x + d]:
                return False
        
        # x is 2nd term: x-d, x+d, x+2d in S
        # d such that x-d >= 1 => d <= x-1
        # and x+2d <= N => 2d <= N-x => d <= (N-x)//2
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x + d] and S[x + 2*d]:
                return False
        
        # x is 1st term: x+d, x+2d, x+3d in S
        # d such that x+3d <= N => 3d <= N-x => d <= (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if S[x + d] and S[x + 2*d] and S[x + 3*d]:
                return False
        
        return True

    # Initial Greedy Construction
    S = bytearray(N + 1) # Indices 0..N, 0 unused
    for x in range(1, N + 1):
        if can_add(x, S):
            S[x] = 1
    
    best_S = bytearray(S)
    best_size = sum(S)
    
    candidates = list(range(1, N + 1))
    
    # Local Search
    while time.time() < deadline:
        # Kick: Remove 1 to 5 random elements
        kick_size = random.randint(1, 5)
        # Get list of elements currently in S
        current_elements = [i for i in range(1, N + 1) if S[i]]
        
        if len(current_elements) > kick_size:
            to_remove = random.sample(current_elements, kick_size)
            S_kicked = bytearray(S)
            for x in to_remove:
                S_kicked[x] = 0
        else:
            S_kicked = bytearray(S)
            
        # Recover: Try to add elements from the complement
        # Randomize the order of candidates to explore different local optima
        # We only need to check elements not in S_kicked
        remaining = [x for x in candidates if not S_kicked[x]]
        random.shuffle(remaining)
        
        S_recovered = S_kicked
        recovered_size = sum(S_recovered) # Start with size of kicked set
        
        for x in remaining:
            if can_add(x, S_recovered):
                S_recovered[x] = 1
                recovered_size += 1
                
        if recovered_size > best_size:
            best_size = recovered_size
            best_S = bytearray(S_recovered)
            # Update S to the new best for the next kick
            S = bytearray(S_recovered)
            
    # Convert bytearray back to sorted list of integers
    return [i for i in range(1, N + 1) if best_S[i]]

import time
import random
import sys

def construct():
    N = 512
    # Safety margin: 85 seconds
    deadline = time.time() + 85.0
    
    # Base-4 construction for initial seed
    def base4_construction():
        S = bytearray(N + 1)
        for mask in range(1, 1 << 9):
            val = 0
            for i in range(9):
                if mask & (1 << i):
                    val += 4 ** i
            if 1 <= val <= N:
                S[val] = 1
        return S

    def can_add(x, S):
        """
        Check if x can be added to set S (represented as bytearray) without creating a 4-AP.
        Returns True if x can be added.
        """
        # x as 1st term: x, x+d, x+2d, x+3d
        # d can be from 1 to (N-x)//3
        for d in range(1, (N - x) // 3 + 1):
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                return False
        
        # x as 2nd term: x-d, x, x+d, x+2d
        # d can be from 1 to min(x-1, (N-x)//2)
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if S[x-d] and S[x+d] and S[x+2*d]:
                return False
        
        # x as 3rd term: x-2d, x-d, x, x+d
        # d can be from 1 to min((x-1)//2, N-x)
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if S[x-2*d] and S[x-d] and S[x+d]:
                return False
        
        # x as 4th term: x-3d, x-2d, x-d, x
        # d can be from 1 to (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if S[x-3*d] and S[x-2*d] and S[x-d]:
                return False
        
        return True

    def get_size(S):
        return sum(S)

    def greedy_fill(S, candidates):
        """
        Try to add candidates to S in the given order.
        """
        for x in candidates:
            if S[x] == 0 and can_add(x, S):
                S[x] = 1

    # Initial best set
    best_S = base4_construction()
    best_size = get_size(best_S)
    
    # List of all elements
    all_elements = list(range(1, N + 1))
    
    # Search Loop
    # We will perform Randomized Local Search (RLS) with kicks.
    # 1. Start from best_S or a random greedy set.
    # 2. Try to add elements.
    # 3. If stuck, kick (remove random elements) and fill again.
    
    # Pre-compute some things? No, just run fast loops.
    
    while time.time() < deadline:
        # Create a copy of the current best to work on
        S = bytearray(best_S)
        current_size = best_size
        
        # Strategy: Randomized Greedy from scratch or from best?
        # Let's try a mix.
        if random.random() < 0.5:
            # Start from base4
            S = bytearray(base4_construction())
            current_size = get_size(S)
        else:
            # Start from empty and greedy
            S = bytearray(N + 1)
            current_size = 0
            random.shuffle(all_elements)
            greedy_fill(S, all_elements)
            current_size = get_size(S)
            
        # Try to improve S via local search / kicks
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing element
            # Shuffle to avoid bias
            missing = [x for x in range(1, N+1) if S[x] == 0]
            random.shuffle(missing)
            for x in missing:
                if can_add(x, S):
                    S[x] = 1
                    current_size += 1
                    improved = True
            
            # 2. If no improvement, try a kick
            if not improved and current_size > 0:
                # Remove 1-3 random elements
                num_remove = random.randint(1, 3)
                present = [x for x in range(1, N+1) if S[x] == 1]
                if len(present) > num_remove:
                    to_remove = random.sample(present, num_remove)
                    for x in to_remove:
                        S[x] = 0
                        current_size -= 1
                    
                    # Try to fill back
                    missing = [x for x in range(1, N+1) if S[x] == 0]
                    random.shuffle(missing)
                    for x in missing:
                        if can_add(x, S):
                            S[x] = 1
                            current_size += 1
                            improved = True
                            
                    if current_size > best_size:
                        best_size = current_size
                        best_S = bytearray(S)
                        improved = True # Keep searching from this better spot? 
                        # Actually, if we improved, we should update best and maybe continue.
                        # But if we just kicked and filled, we might be stuck again.
                        # Let's mark improved to allow another round of adding.
                        # But we need to break out of the inner while if we want to try a new restart?
                        # No, the inner while is "improvement loop". 
                        # If we kick and fill, and size increases, it's an improvement.
                        # If size doesn't increase, we loop again?
                        # The `improved` flag was False. We kicked. 
                        # If current_size > best_size, we update best. 
                        # Should we continue the inner loop? 
                        # If we updated best, we are in a better state. We should try to add more.
                        # So setting improved = True is correct to allow another pass of "add missing".
                        # But if we don't improve size, we are stuck. 
                        # The outer while will restart.
        
        # If current solution is better, update global best
        if current_size > best_size:
            best_size = current_size
            best_S = bytearray(S)

    # Convert bytearray to list of ints for return
    result = [i for i in range(1, N + 1) if best_S[i]]
    return result


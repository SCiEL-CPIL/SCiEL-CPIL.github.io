import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 5s safety margin
    
    # Helper to check if a set of sums is valid (distinct)
    # We maintain S as a set of sums.
    
    def get_initial_solution():
        A = set()
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            # Check if x can be added
            # New sums: x+a for a in A, and 2x
            # We need to ensure no conflict with existing S and no internal conflict
            # Internal conflict: x+a1 == x+a2 => a1==a2 (impossible since A is set)
            # x+a == 2x => a==x (impossible since x not in A yet)
            # So we just check intersection with S.
            # Wait, what if x+a1 == x+a2? No, a1!=a2.
            # What if x+a == 2x? a=x. x not in A.
            # So new sums are distinct internally.
            # Just check intersection with S.
            
            # Optimization: generate sums, check intersection
            new_sums = [x + a for a in A]
            new_sums.append(2 * x)
            
            if not set(new_sums).intersection(S):
                A.add(x)
                S.update(new_sums)
        return A, S

    def is_valid(A, S):
        # S is assumed to be the set of sums of A.
        # Check if S is consistent with A? 
        # We maintain S incrementally, so it should be.
        # But we need to check Sidon property: all sums distinct.
        # If we maintained S as a set, and we only added distinct sums, it's valid.
        # However, if we do a swap, we must be careful.
        # Let's assume S is correct for A.
        # We need to check Maximality.
        
        # Count unblocked elements
        unblocked = 0
        for x in range(1, N + 1):
            if x in A:
                continue
            # Check if x is blocked
            # x is blocked if exists a in A such that x+a in S
            blocked = False
            for a in A:
                if x + a in S:
                    blocked = True
                    break
            if not blocked:
                unblocked += 1
        
        return unblocked == 0, unblocked

    def energy(A, S):
        # Check Sidon validity by checking if S size matches expected?
        # Expected size k(k+1)/2.
        # If we messed up S, this might fail.
        # But we update S carefully.
        # Let's do a quick check: if len(S) != len(A)*(len(A)+1)//2: return 10000
        k = len(A)
        expected_sums = k * (k + 1) // 2
        if len(S) != expected_sums:
            return 10000 + (len(S) - expected_sums)
        
        is_max, unblocked = is_valid(A, S)
        if not is_max:
            return 1000 + k + unblocked * 10
        return k

    # 1. Get initial best solution via multiple random greedy runs
    best_A, best_S = None, None
    best_energy = float('inf')
    
    runs = 50
    for i in range(runs):
        if time.time() > deadline: break
        A, S = get_initial_solution()
        e = energy(A, S)
        if e < best_energy:
            best_energy = e
            best_A = set(A)
            best_S = set(S)

    if best_A is None:
        # Fallback
        A, S = get_initial_solution()
        return list(A)

    # 2. Simulated Annealing
    current_A = best_A
    current_S = best_S
    current_energy = best_energy
    
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        new_A = set(current_A)
        new_S = set(current_S)
        
        if move_type < 0.3 and current_A:
            # Remove
            x = random.choice(list(current_A))
            # Remove sums involving x
            # Sums are x+a for a in A, including 2x (a=x)
            to_remove = [x + a for a in current_A]
            for s in to_remove:
                new_S.discard(s)
            new_A.discard(x)
            
        elif move_type < 0.6 and len(current_A) < N:
            # Add
            # Pick random y not in A
            # To make it faster, pick from a small subset or random
            # If A is small, N-|A| is large.
            # Just pick random from 1..N
            y = random.randint(1, N)
            if y in current_A:
                continue # Retry
            # Check if y can be added to current_A
            # If it conflicts, it's an invalid move, but SA can handle high energy
            # However, we must update S correctly for the energy calculation
            # If we add y, new sums are y+a for a in current_A, and 2y
            # We must check if these conflict with current_S.
            # If they do, the new state is invalid (Sidon violation).
            # We can still calculate energy (it will be high).
            # But we must add them to new_S to reflect the state.
            new_sums = [y + a for a in current_A]
            new_sums.append(2 * y)
            # Add to new_S
            # Note: if there are conflicts, len(new_S) will be less than expected for the new size
            for s in new_sums:
                new_S.add(s)
            new_A.add(y)
            
        else:
            # Swap
            if not current_A or len(current_A) == N:
                continue
            x = random.choice(list(current_A))
            y = random.randint(1, N)
            if y in current_A:
                continue
            
            # Remove x
            to_remove = [x + a for a in current_A]
            for s in to_remove:
                new_S.discard(s)
            new_A.discard(x)
            
            # Add y to the NEW A (which is old A - x)
            # Sums involving y with new_A
            new_sums = [y + a for a in new_A]
            new_sums.append(2 * y)
            for s in new_sums:
                new_S.add(s)
            new_A.add(y)

        new_energy = energy(new_A, new_S)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_A = new_A
            current_S = new_S
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_A = set(current_A)
                best_S = set(current_S)
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return the best solution found
    # The problem asks for a list of integers
    return list(best_A)

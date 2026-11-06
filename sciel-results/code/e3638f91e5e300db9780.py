import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    deadline = time.time() + 80  # 80s to be safe, limit is 90s
    
    # Precomputation
    # For each x, store lists of tuples of other indices needed to form APs with x
    # 3-APs:
    # x is start: (x+d, x+2d)
    # x is mid: (x-d, x+d)
    # x is end: (x-2d, x-d)
    # 4-APs:
    # x is start: (x+d, x+2d, x+3d)
    # x is 2nd: (x-d, x+d, x+2d)
    # x is 3rd: (x-2d, x-d, x+d)
    # x is end: (x-3d, x-2d, x-d)
    
    starts_3 = [[] for _ in range(n+1)]
    mids_3 = [[] for _ in range(n+1)]
    ends_3 = [[] for _ in range(n+1)]
    
    starts_4 = [[] for _ in range(n+1)]
    mids_4a = [[] for _ in range(n+1)] # x is 2nd
    mids_4b = [[] for _ in range(n+1)] # x is 3rd
    ends_4 = [[] for _ in range(n+1)]
    
    for x in range(1, n+1):
        # 3-APs
        # x start: x+d, x+2d <= n => d <= (n-x)//2
        for d in range(1, (n-x)//2 + 1):
            starts_3[x].append((x+d, x+2*d))
        # x mid: x-d >= 1, x+d <= n => d <= min(x-1, n-x)
        for d in range(1, min(x-1, n-x) + 1):
            mids_3[x].append((x-d, x+d))
        # x end: x-2d >= 1 => d <= (x-1)//2
        for d in range(1, (x-1)//2 + 1):
            ends_3[x].append((x-2*d, x-d))
            
        # 4-APs
        # x start: x+3d <= n => d <= (n-x)//3
        for d in range(1, (n-x)//3 + 1):
            starts_4[x].append((x+d, x+2*d, x+3*d))
        # x 2nd: x-d >= 1, x+2d <= n => d <= min(x-1, (n-x)//2)
        for d in range(1, min(x-1, (n-x)//2) + 1):
            mids_4a[x].append((x-d, x+d, x+2*d))
        # x 3rd: x-2d >= 1, x+d <= n => d <= min((x-1)//2, n-x)
        for d in range(1, min((x-1)//2, n-x) + 1):
            mids_4b[x].append((x-2*d, x-d, x+d))
        # x end: x-3d >= 1 => d <= (x-1)//3
        for d in range(1, (x-1)//3 + 1):
            ends_4[x].append((x-3*d, x-2*d, x-d))

    # State representation: boolean numpy array
    # A[i] is True if i is in the set
    
    def count_3aps(A):
        count = 0
        for x in range(1, n+1):
            if A[x]:
                for y, z in starts_3[x]:
                    if A[y] and A[z]:
                        count += 1
                for y, z in mids_3[x]:
                    if A[y] and A[z]:
                        count += 1
                for y, z in ends_3[x]:
                    if A[y] and A[z]:
                        count += 1
        return count // 3 # Each 3AP is counted 3 times (once for each member as start/mid/end? No, wait.)
        # Let's re-verify counting.
        # If A={1,2,3}, 3AP is (1,2,3).
        # x=1 (start): (2,3) in A? Yes. Count +1.
        # x=2 (mid): (1,3) in A? Yes. Count +1.
        # x=3 (end): (1,2) in A? Yes. Count +1.
        # Total 3. So divide by 3. Correct.

    def count_4aps(A):
        count = 0
        for x in range(1, n+1):
            if A[x]:
                for y, z, w in starts_4[x]:
                    if A[y] and A[z] and A[w]:
                        count += 1
                for y, z, w in mids_4a[x]:
                    if A[y] and A[z] and A[w]:
                        count += 1
                for y, z, w in mids_4b[x]:
                    if A[y] and A[z] and A[w]:
                        count += 1
                for y, z, w in ends_4[x]:
                    if A[y] and A[z] and A[w]:
                        count += 1
        return count // 4

    def delta_3ap(x, A, is_add):
        """Calculate change in 3AP count when adding/removing x."""
        delta = 0
        if is_add:
            # New 3APs formed
            for y, z in starts_3[x]:
                if A[y] and A[z]:
                    delta += 1
            for y, z in mids_3[x]:
                if A[y] and A[z]:
                    delta += 1
            for y, z in ends_3[x]:
                if A[y] and A[z]:
                    delta += 1
        else:
            # 3APs broken
            for y, z in starts_3[x]:
                if A[y] and A[z]:
                    delta -= 1
            for y, z in mids_3[x]:
                if A[y] and A[z]:
                    delta -= 1
            for y, z in ends_3[x]:
                if A[y] and A[z]:
                    delta -= 1
        return delta

    def delta_4ap(x, A, is_add):
        """Calculate change in 4AP count when adding/removing x."""
        delta = 0
        if is_add:
            for y, z, w in starts_4[x]:
                if A[y] and A[z] and A[w]:
                    delta += 1
            for y, z, w in mids_4a[x]:
                if A[y] and A[z] and A[w]:
                    delta += 1
            for y, z, w in mids_4b[x]:
                if A[y] and A[z] and A[w]:
                    delta += 1
            for y, z, w in ends_4[x]:
                if A[y] and A[z] and A[w]:
                    delta += 1
        else:
            for y, z, w in starts_4[x]:
                if A[y] and A[z] and A[w]:
                    delta -= 1
            for y, z, w in mids_4a[x]:
                if A[y] and A[z] and A[w]:
                    delta -= 1
            for y, z, w in mids_4b[x]:
                if A[y] and A[z] and A[w]:
                    delta -= 1
            for y, z, w in ends_4[x]:
                if A[y] and A[z] and A[w]:
                    delta -= 1
        return delta

    # Initial State: Greedy
    A = np.zeros(n+1, dtype=bool)
    for x in range(1, n+1):
        # Check if adding x creates a 4AP
        if delta_4ap(x, A, True) == 0:
            # Also check if it adds 3APs? Or just add?
            # Let's just add if no 4AP is created.
            A[x] = True
            
    # Calculate initial score
    score = count_3aps(A)
    violations = count_4aps(A)
    
    best_score = score
    best_A = A.copy()
    
    # Parallel Tempering
    # Number of chains
    num_chains = 10
    # Temperatures: geometric progression
    T_min = 0.1
    T_max = 10.0
    temperatures = np.geomspace(T_min, T_max, num_chains)
    
    # Initialize chains
    chains = []
    for i in range(num_chains):
        # Start each chain with a perturbed version of the best state
        A_chain = best_A.copy()
        # Randomly flip a few bits
        for _ in range(10):
            x = random.randint(1, n)
            A_chain[x] = not A_chain[x]
        # Ensure validity? No, let SA handle it.
        # But we need to track score and violations.
        s = count_3aps(A_chain)
        v = count_4aps(A_chain)
        chains.append({
            'A': A_chain,
            'score': s,
            'viol': v,
            'energy': -s + 100 * v # Penalty for violations
        })
    
    # Update best if initial greedy was better
    # (It should be, as it's valid)
    
    iterations = 0
    swap_interval = 500
    
    while time.time() < deadline:
        for i in range(num_chains):
            if time.time() > deadline - 2:
                break
            chain = chains[i]
            A_chain = chain['A']
            
            # Pick a random move
            x = random.randint(1, n)
            is_add = not A_chain[x]
            
            d_score = delta_3ap(x, A_chain, is_add)
            d_viol = delta_4ap(x, A_chain, is_add)
            
            new_score = chain['score'] + d_score
            new_viol = chain['viol'] + d_viol
            new_energy = -new_score + 100 * new_viol
            old_energy = chain['energy']
            
            delta_E = new_energy - old_energy
            
            # Metropolis criterion
            if delta_E < 0:
                accept = True
            else:
                # Accept with probability exp(-delta_E / T)
                # To avoid overflow/underflow issues, just check log
                # T is temperatures[i]
                T = temperatures[i]
                # If delta_E is large and T is small, prob is 0
                if delta_E / T > 50: # exp(-50) is tiny
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta_E / T)
            
            if accept:
                A_chain[x] = is_add
                chain['score'] = new_score
                chain['viol'] = new_viol
                chain['energy'] = new_energy
                
                # Update global best if valid
                if new_viol == 0 and new_score > best_score:
                    best_score = new_score
                    best_A = A_chain.copy()
            
            iterations += 1
            if iterations % swap_interval == 0:
                # Swap adjacent chains
                for j in range(num_chains - 1):
                    c1 = chains[j]
                    c2 = chains[j+1]
                    T1 = temperatures[j]
                    T2 = temperatures[j+1]
                    
                    # Probability of swap: min(1, exp((E1/T2 - E2/T1 - E1/T1 + E2/T2)))
                    # = min(1, exp((E1*(1/T2 - 1/T1) - E2*(1/T1 - 1/T2))))
                    # Let's use standard formula:
                    # P = min(1, exp( - (E1/T1 - E2/T1 - E1/T2 + E2/T2) ) )
                    # = min(1, exp( (E2 - E1)/T1 - (E2 - E1)/T2 ) )
                    # = min(1, exp( (E2 - E1) * (1/T1 - 1/T2) ) )
                    
                    delta_swap = (c2['energy'] - c1['energy']) * (1.0/T1 - 1.0/T2)
                    if delta_swap < 0 or random.random() < np.exp(delta_swap):
                        # Swap states
                        chains[j], chains[j+1] = chains[j+1], chains[j]
        
        if time.time() > deadline - 2:
            break

    # Return the best valid set found
    return [i for i in range(1, n+1) if best_A[i]]
# EVOLVE-BLOCK-END

from math import gcd
import random

# Precompute GCD matrix for 1..128
N = 128
gcd_mat = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i, N+1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

def can_add(x, S, gcd_mat):
    """
    Check if x can be added to S without creating a bad triple.
    A bad triple (a, b, x) has gcd(a,b) == gcd(a,x) == gcd(b,x).
    """
    n = len(S)
    # Optimization: Group elements in S by their gcd with x?
    # Or just simple double loop which is fast enough for N=128 and |S|~50-80.
    # 80^2 = 6400 checks. 128 * 6400 = 800k ops. Very fast.
    for i in range(n):
        a = S[i]
        g_ax = gcd_mat[a][x]
        for j in range(i+1, n):
            b = S[j]
            if gcd_mat[a][b] == g_ax and gcd_mat[b][x] == g_ax:
                return False
    return True

def local_improve(S, gcd_mat):
    S_list = list(S)
    S_set = set(S_list)
    improved = True
    iterations = 0
    max_iterations = 100 # Increased limit
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # 1. Try to add any single element
        for x in range(1, N+1):
            if x not in S_set and can_add(x, S_list, gcd_mat):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try 1-for-1 swap
        # Sample a few removal candidates
        if len(S_list) > 0:
            # Shuffle S_list to pick random candidates
            candidates_to_remove = random.sample(S_list, min(10, len(S_list)))
            for x in candidates_to_remove:
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                # Try to add any y
                for y in range(1, N+1):
                    if y in S_temp_set:
                        continue
                    if can_add(y, S_temp_list, gcd_mat):
                        S_list = S_temp_list + [y]
                        S_set = S_temp_set | {y}
                        improved = True
                        break
                if improved:
                    break

        if improved:
            continue

        # 3. Try 1-for-2 swap (Remove 1, Add 2) - Net gain +1
        if len(S_list) > 0:
            candidates_to_remove = random.sample(S_list, min(10, len(S_list)))
            for x in candidates_to_remove:
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                added = 0
                new_S_list = list(S_temp_list)
                new_S_set = set(S_temp_set)
                
                # Try to add 2 elements
                candidates = list(range(1, N+1))
                random.shuffle(candidates)
                for y in candidates:
                    if y in new_S_set:
                        continue
                    if can_add(y, new_S_list, gcd_mat):
                        new_S_list.append(y)
                        new_S_set.add(y)
                        added += 1
                        if added == 2:
                            break
                            
                if added == 2:
                    S_list = new_S_list
                    S_set = new_S_set
                    improved = True
                    break

        if improved:
            continue

        # 4. Try 2-for-1 swap (Remove 2, Add 1) - Net gain -1, but might unlock future adds
        # This is rarely useful for maximization unless stuck in a local minimum that blocks a big add.
        # Given time constraints, we skip this or do it sparingly.
        
    return S_list

def construct():
    best_set = []
    
    # Heuristic orders:
    # 1. Natural order
    # 2. Reverse order
    # 3. Order by number of divisors (ascending) - Primes first?
    # 4. Random shuffles
    
    def count_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = [count_divisors(i) for i in range(N+1)]
    order_by_div = list(range(1, N+1))
    order_by_div.sort(key=lambda x: div_counts[x])
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        order_by_div,
        list(reversed(order_by_div)),
    ]
    
    # Add random orders
    for _ in range(100):
        cand = list(range(1, 129))
        random.shuffle(cand)
        orders.append(cand)
        
    for order in orders:
        S = []
        for x in order:
            if can_add(x, S, gcd_mat):
                S.append(x)
                
        # Local improvement
        S = local_improve(S, gcd_mat)
        
        if len(S) > len(best_set):
            best_set = S
            
    return best_set

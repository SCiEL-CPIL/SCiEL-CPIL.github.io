import time
import random
from itertools import combinations

# Precomputation
N = 128
# conflicts[x] will store a list of pairs (a, b) such that {x, a, b} is a forbidden triple.
# Note: a < b < x or similar to avoid duplicates? No, just store unique pairs.
conflicts = [[] for _ in range(N + 1)]

def precompute_conflicts():
    # We need to find all triples (a, b, c) such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    # To speed up, we can iterate over possible LCM values L.
    # For each L, find all divisors of L.
    # For each pair of divisors (d1, d2) of L, if lcm(d1, d2) == L, then they are "L-compatible".
    # If we have a set of numbers D_L such that pairwise lcm is L, any 3 of them form a forbidden triple.
    
    # Actually, simpler: just iterate a, b, c? 128^3 is ~2M. Python might be slow for 2M lcm calls.
    # 2M operations in Python is ~0.1-0.5s. Acceptable.
    
    # Optimization: 
    # lcm(a,b) = L.
    # We can iterate a from 1 to 128.
    # b from a+1 to 128.
    # L = lcm(a,b).
    # If L > 128, skip? No, L can be > 128. But c must be <= 128.
    # If L > 128, can lcm(a,c) be L? Yes, if c is a divisor of L and lcm(a,c)=L.
    # But if L > 128, and c <= 128, then c must divide L.
    
    # Let's just use the direct check with caching or simple math.
    # math.lcm is fast.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # If L_ab is very large, it's unlikely to be matched by c <= 128?
            # lcm(a,c) = L_ab implies c must be a multiple of L_ab / gcd(a, L_ab)?
            # Actually, c must be a divisor of L_ab? No.
            # lcm(a,c) = L implies L is a multiple of a and c.
            # So c must divide L_ab.
            # So we only need to check c that are divisors of L_ab.
            
            # Find divisors of L_ab that are <= N
            divisors = []
            # Simple divisor finding
            limit = int(L_ab**0.5)
            for i in range(1, limit + 1):
                if L_ab % i == 0:
                    if i <= N:
                        divisors.append(i)
                    other = L_ab // i
                    if other != i and other <= N:
                        divisors.append(other)
            
            # For each c in divisors, c != a, c != b
            for c in divisors:
                if c == a or c == b:
                    continue
                # Check if lcm(a,c) == L_ab and lcm(b,c) == L_ab
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    # Triple (a, b, c) is forbidden
                    # Add to conflicts
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))

# Import math inside function to keep global clean, or just import at top.
import math

# Run precomputation once?
# The task says "Keep the function names and signatures".
# We can do precomputation in a global variable or inside construct with a cache.
# Since construct is called once per evaluation, global is fine.

# Initialize
precompute_conflicts()

def can_add(x, in_set):
    """Check if x can be added to the set defined by in_set."""
    # in_set is a list of booleans, index 1..N
    for a, b in conflicts[x]:
        if in_set[a] and in_set[b]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = [False] * (N + 1)
    best_size = 0
    
    # Initial greedy
    order = list(range(1, N + 1))
    random.shuffle(order)
    current_set = [False] * (N + 1)
    current_size = 0
    for x in order:
        if can_add(x, current_set):
            current_set[x] = True
            current_size += 1
    
    if current_size > best_size:
        best_size = current_size
        best_set = current_set[:]
    
    # Local Search with Restarts
    # We will try to improve the current set, and if stuck, restart with a new random greedy.
    
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any element
            # To make it faster, we can iterate over a random subset of candidates
            candidates = [x for x in range(1, N + 1) if not current_set[x]]
            random.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if can_add(x, current_set):
                    current_set[x] = True
                    current_size += 1
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # 2. Try a swap: remove one, add one or more
            # Pick a random element in the set to remove
            in_elements = [x for x in range(1, N + 1) if current_set[x]]
            if not in_elements:
                break
            
            u = random.choice(in_elements)
            current_set[u] = False
            current_size -= 1
            
            # Try to add as many as possible from candidates
            # We prioritize elements that were blocked by u?
            # Just try all candidates
            candidates = [x for x in range(1, N + 1) if not current_set[x]]
            random.shuffle(candidates)
            
            added_count = 0
            for x in candidates:
                if can_add(x, current_set):
                    current_set[x] = True
                    current_size += 1
                    added_count += 1
                    # We can stop after adding 1 if we just want to maintain size, 
                    # but adding more is better.
                    # However, adding too many might block future better additions?
                    # For now, greedily add all possible? No, that's expensive.
                    # Let's just add 1 to keep it simple and fast, or break after 1.
                    # Actually, if we remove 1, we should try to add 2 to gain size.
                    # So we should try to add as many as possible.
                    # But checking can_add is O(degree).
                    # Let's just try to add one. If we add one, size is same. 
                    # If we add two, size increases.
                    # To maximize chance of increase, we should try to add multiple.
                    # But if we add one, we might block the second.
                    # Greedy addition is fine.
                    
            if added_count > 0:
                improved = True
                # Update best if better
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
            else:
                # Restore u
                current_set[u] = True
                current_size += 1
        
        # If we exited the inner loop without improving (or time up), restart
        # Create a new random greedy solution
        order = list(range(1, N + 1))
        random.shuffle(order)
        current_set = [False] * (N + 1)
        current_size = 0
        for x in order:
            if can_add(x, current_set):
                current_set[x] = True
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
    # Convert boolean array to list of ints
    result = [i for i in range(1, N + 1) if best_set[i]]
    
    # Ensure nonempty
    if not result:
        result = [1]
        
    return result

# Ensure math is imported
import math

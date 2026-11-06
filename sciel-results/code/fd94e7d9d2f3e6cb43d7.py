import time

def is_square_free(n):
    # Check if n is square-free
    # n is square-free if no square of a prime divides n
    # For n up to 10^6 (since b^3 <= 10^12 -> b <= 10^4), this is fast
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def generate_powerful(limit):
    powerful = set([1])
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(limit ** (1/3))
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        # a^2 * b^3 <= limit => a^2 <= limit / b^3 => a <= sqrt(limit / b^3)
        max_a = int((limit // b3) ** 0.5)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerful.add(n)
    return sorted(list(powerful))

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 20s buffer
    
    # Generate powerful numbers up to 10^12
    limit = 10**12
    P = generate_powerful(limit)
    
    # Count sums
    counts = {}
    n = len(P)
    for i in range(n):
        a = P[i]
        if a > limit: break # P is sorted
        for j in range(i, n):
            b = P[j]
            s = a + b
            if s > limit:
                break
            if s in counts:
                counts[s] += 1
            else:
                counts[s] = 1
                
    # Find max
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Construct pairs
    P_set = set(P)
    pairs = []
    for a in P:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}

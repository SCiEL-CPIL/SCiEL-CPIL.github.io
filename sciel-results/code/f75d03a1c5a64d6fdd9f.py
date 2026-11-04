import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Use the standard Sidon set construction: U = {i*M + i^2 : i=0,...,n-1}
    # with M > 2n ensures all pairwise sums are distinct
    # |U| = n, max(U) = (n-1)*M + (n-1)^2
    # |U+U| = n(n+1)/2, |U-U| = n^2
    # ratio = 2n/(n+1)
    # score = 1 + log(2n/(n+1)) / log(2*((n-1)*M + (n-1)^2) + 1)
    
    # To maximize score, we want large n and small max(U)
    # With M = 2n+1: max(U) = (n-1)(2n+1) + (n-1)^2 = (n-1)(2n+1+n-1) = (n-1)(3n) = 3n^2 - 3n
    # log(2*max(U)+1) ≈ log(6n^2) ≈ 2*log(n) + log(6)
    # log(2n/(n+1)) ≈ log(2) - log(1+1/n) ≈ log(2) for large n
    # score ≈ 1 + log(2) / (2*log(n) + log(6))
    
    # Try different n values and pick the best
    best_U = [0]
    best_score = 1.0
    
    for n in range(2, 5000):
        M = 2 * n + 1
        U = [i * M + i * i for i in range(n)]
        max_U = U[-1]
        
        # Compute |U+U| and |U-U|
        # For this Sidon construction, we know:
        # |U+U| = n(n+1)/2
        # |U-U| = n^2
        # But let's verify for small n and use formula for large n
        
        if n <= 100:
            sum_set = set()
            diff_set = set()
            for a in U:
                for b in U:
                    sum_set.add(a + b)
                    diff_set.add(a - b)
            sum_size = len(sum_set)
            diff_size = len(diff_set)
        else:
            sum_size = n * (n + 1) // 2
            diff_size = n * n
        
        if sum_size == 0:
            continue
        
        import math
        ratio = diff_size / sum_size
        if ratio <= 1:
            continue
        
        denom = math.log(2 * max_U + 1)
        if denom <= 0:
            continue
        
        score = 1 + math.log(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_U = U
        
        if time.time() > deadline:
            break
    
    return best_U

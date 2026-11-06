    # Local Search on best_S
    if best_S:
        S = best_S
        S_set = set(S)
        excluded = [x for x in range(N + 1) if x not in S_set]
        
        # Shuffle excluded for randomness
        random.shuffle(excluded)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add each excluded element
            # This is O(N * P) per element. 5000 * 600 = 3M.
            # 5000 elements * 3M = 15B ops. Too slow.
            
            # We need a faster check.
            # Let's just try a random subset of excluded elements.
            for y in excluded[:100]: # Try first 100
                if y in S_set: continue
                
                # Check if S U {y} is admissible
                # We need to check primes <= len(S) + 1
                new_k = len(S) + 1
                valid = True
                for p in all_primes:
                    if p > new_k: break
                    counts = [0] * p
                    for x in S:
                        counts[x % p] += 1
                    counts[y % p] += 1
                    if min(counts) > 0:
                        valid = False
                        break
                if valid:
                    S.append(y)
                    S_set.add(y)
                    excluded.remove(y)
                    improved = True
                    # Restart loop to try more additions
                    break
                    
    return S

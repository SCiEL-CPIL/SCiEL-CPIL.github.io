import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)
    
    def tau_perp(n, prime_factors, exponents):
        divs = [1]
        for p, e in zip(prime_factors, exponents):
            new_divs = []
            pow_p = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * pow_p)
                pow_p *= p
            divs = new_divs
        divs.sort()
        score = 0
        gcd = math.gcd
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def calc_n(prime_factors, exponents):
        n = 1
        for p, e in zip(prime_factors, exponents):
            n *= p ** e
        return n

    best_n = 2
    best_score = 1

    # Generate HCN candidates
    # We will iterate k from 1 to 15
    for k in range(1, 16):
        if time.time() > deadline - 20.0:
            return best_n
            
        p_set = primes[:k]
        # Start with e=[1]*k
        e = [1] * k
        n = calc_n(p_set, e)
        
        if n > 10**18:
            continue
            
        # Store this candidate
        candidates = []
        
        # Greedy HCN generation
        current_e = e[:]
        current_n = n
        
        # Add initial candidate
        candidates.append((current_n, current_e[:]))
        
        # Iteratively increase exponents
        while True:
            # Find the index with min exponent
            min_exp = min(current_e)
            # Find all indices with this min exponent
            candidates_idx = [i for i in range(k) if current_e[i] == min_exp]
            
            if not candidates_idx:
                break
                
            # Try each candidate index
            found_next = False
            for i in candidates_idx:
                new_n = current_n * p_set[i]
                if new_n <= 10**18:
                    # This is a valid HCN step
                    new_e = current_e[:]
                    new_e[i] += 1
                    candidates.append((new_n, new_e))
                    # We should explore this branch?
                    # HCNs form a tree. The "main" HCN chain is the one with max tau.
                    # But we might miss good coprime structures in branches.
                    # However, the number of HCNs is small.
                    # Let's just follow the main chain (smallest p for tie-break? No, any is fine for max tau).
                    # Actually, for max tau, we just need to pick ONE.
                    # But to be safe, we can add all valid neighbors to the candidate list.
                    found_next = True
                    break # Pick the first valid one to continue the chain? 
                    # Or add all and then pick the best later?
                    # Adding all is better for search.
            
            # If we found a next step in the main chain, update current state to continue
            # But we added all valid next steps to candidates.
            # To continue the greedy chain, we pick the one that gives max tau (which is any of them).
            # Let's pick the one with smallest prime to keep numbers small?
            # Actually, it doesn't matter for tau.
            # Let's just pick the first valid one to update current_n and current_e.
            if found_next:
                # Re-find the specific one we picked
                min_exp = min(current_e)
                for i in range(k):
                    if current_e[i] == min_exp:
                        test_n = current_n * p_set[i]
                        if test_n <= 10**18:
                            current_n = test_n
                            current_e[i] += 1
                            break
            else:
                break
                
        # Now we have a list of candidates for this k
        # Evaluate them
        for n_cand, e_cand in candidates:
            if time.time() > deadline - 10.0:
                return best_n
            s = tau_perp(n_cand, p_set, e_cand)
            if s > best_score:
                best_score = s
                best_n = n_cand
                
        # Local search around the best candidate for this k
        if candidates:
            # Find the best candidate in this list
            best_cand_score = -1
            best_cand_n = 0
            best_cand_e = None
            for n_cand, e_cand in candidates:
                s = tau_perp(n_cand, p_set, e_cand)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand_n = n_cand
                    best_cand_e = e_cand[:]
                    
            # Hill climbing from best_cand
            current_n = best_cand_n
            current_e = best_cand_e
            current_score = best_cand_score
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Try all single-step moves
                best_move_score = current_score
                best_move_n = current_n
                best_move_e = current_e[:]
                
                for i in range(k):
                    # Increase
                    new_n = current_n * p_set[i]
                    if new_n <= 10**18:
                        new_e = current_e[:]
                        new_e[i] += 1
                        s = tau_perp(new_n, p_set, new_e)
                        if s > best_move_score:
                            best_move_score = s
                            best_move_n = new_n
                            best_move_e = new_e
                            
                    # Decrease
                    if current_e[i] > 1:
                        new_n = current_n // p_set[i]
                        new_e = current_e[:]
                        new_e[i] -= 1
                        s = tau_perp(new_n, p_set, new_e)
                        if s > best_move_score:
                            best_move_score = s
                            best_move_n = new_n
                            best_move_e = new_e
                            
                if best_move_score > current_score:
                    current_n = best_move_n
                    current_e = best_move_e
                    current_score = best_move_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                else:
                    # Try some random moves to escape local maxima
                    for _ in range(5):
                        if time.time() > deadline - 5.0:
                            return best_n
                        # Random decrease
                        dec_idx = random.randint(0, k-1)
                        if current_e[dec_idx] > 1:
                            new_n = current_n // p_set[dec_idx]
                            new_e = current_e[:]
                            new_e[dec_idx] -= 1
                            # Fill up with increases
                            # Greedy fill
                            for j in range(k):
                                while new_n * p_set[j] <= 10**18:
                                    new_n *= p_set[j]
                                    new_e[j] += 1
                            if new_n <= 10**18:
                                s = tau_perp(new_n, p_set, new_e)
                                if s > best_move_score:
                                    best_move_score = s
                                    best_move_n = new_n
                                    best_move_e = new_e
                                    improved = True
                                    if s > best_score:
                                        best_score = s
                                        best_n = new_n
                    if not improved:
                        break

    return best_n

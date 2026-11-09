import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210
    target = masks[210]

    # Basis: dict mapping highest-bit position -> (vector, history_mask)
    basis = {}
    basis_hist = {}

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # history bit for this candidate

        # Insert v into basis with history h
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                tv ^= basis[p]
                th ^= basis_hist[p]
            else:
                basis[p] = tv
                basis_hist[p] = th
                break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                tv ^= basis[p]
                th ^= basis_hist[p]
            else:
                break

        if tv == 0:
            # Extract solution from history mask th
            sol_set = set()
            hh = th
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_set.add(211 + idx)
                hh ^= lsb

            if len(sol_set) <= 100:
                sol = list(sol_set)
                # Verify the product condition
                prod = 210
                for x in sol:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(x) for x in sol]
            else:
                # Attempt to reduce solution size using kernel vectors
                # 1. Find pairs (a, b) with masks[a] == masks[b]
                #    If both in sol, remove both.
                #    If one in sol, one not, swap (no size change, but might help later?)
                #    Actually, swapping doesn't change size. Only removing helps.
                #    So we look for pairs where both are in sol_set.
                
                # 2. Find triples (a, b, c) with masks[a] ^ masks[b] ^ masks[c] == 0
                #    If all 3 in sol, remove all 3.
                #    If 2 in sol, 1 not, remove 2, add 1 (size -1).
                #    If 1 in sol, 2 not, remove 1, add 2 (size +1, bad).
                #    If 0 in sol, add 3 (size +3, bad).
                
                # Let's collect dependencies
                # This is expensive if M is large. M is at most ~230-250 for score 22.5?
                # Score 22.5 means max - 210 = 22.5? No, score = seed / value.
                # If seed is 1, value = 22.5? No, value is integer.
                # The prompt says "score 22.5000".
                # If the metric is 1/value, and score is 22.5, then value is 1/22.5? No.
                # "search score is seed/value". If seed=1, score=1/value.
                # If score=22.5, value=1/22.5 = 0.044? Impossible for integer difference.
                # Maybe the score is defined differently in the prompt's context or I am misinterpreting "seed/value".
                # "The evaluator measures the raw value (lower is better). The search score is seed/value".
                # If raw value is $V$, score is $S/V$.
                # If current score is 22.5, and $V$ is an integer $\ge 1$, then $S$ must be large.
                # Or maybe $V$ is small. If $V=1$, score=22.5 implies seed=22.5.
                # Let's assume the goal is to minimize $V = \max(S) - 210$.
                # The current program finds the minimal $M$ such that a solution exists.
                # $V = M - 210$.
                # If the solution found at minimal $M$ has $>100$ elements, we must go to $M+1$.
                # This increases $V$ by 1, decreasing the score.
                # So reducing the size of the solution at minimal $M$ is crucial to avoid this penalty.
                
                # Heuristic reduction:
                # Map mask to list of indices in {211..M}
                mask_to_indices = {}
                for i in range(211, M + 1):
                    m = masks[i]
                    if m not in mask_to_indices:
                        mask_to_indices[m] = []
                    mask_to_indices[m].append(i)
                
                # Try to remove pairs
                changed = True
                while changed and len(sol_set) > 100:
                    changed = False
                    for m, indices in mask_to_indices.items():
                        if m == 0:
                            continue # Zero mask elements don't help with XOR, but if in sol, they can be removed
                            # Wait, if mask is 0, it's a dependency by itself? No, v_i=0 means it's a square.
                            # If v_i=0, it can be in or out of sol without changing the XOR.
                            # So if i is in sol_set, we can remove it.
                            for i in indices:
                                if i in sol_set:
                                    sol_set.remove(i)
                                    changed = True
                        else:
                            # Check for pairs in sol_set
                            in_sol = [i for i in indices if i in sol_set]
                            if len(in_sol) >= 2:
                                # Remove two
                                sol_set.remove(in_sol[0])
                                sol_set.remove(in_sol[1])
                                changed = True
                                break # Restart loop to be safe
                    if changed:
                        break # Restart while loop to re-evaluate
                
                # Try to use triples (if still > 100)
                # This is O(N^3) or O(N^2) with hash map.
                # If len(sol_set) > 100, try to find triples that reduce size.
                if len(sol_set) > 100:
                    # Find triples (a,b,c) with xor 0
                    # Iterate over pairs (a,b) in sol_set, check if mask[a]^mask[b] is in mask_to_indices
                    # and if the third element c is NOT in sol_set (to reduce size by 1: remove a,b add c? No, remove a,b,c? No.)
                    # If a,b in sol, c not in sol: new sol = sol \ {a,b} \ {c}? No.
                    # Dependency: v_a ^ v_b ^ v_c = 0.
                    # Current XOR: T.
                    # If a,b in sol, c not in sol:
                    #   Remove a,b from sol: XOR becomes T ^ v_a ^ v_b.
                    #   We need to add c? T ^ v_a ^ v_b = T ^ (v_a ^ v_b) = T ^ v_c (since v_a^v_b^v_c=0 => v_a^v_b=v_c).
                    #   So new XOR is T ^ v_c. This is NOT T.
                    #   Wait. If we replace {a,b} with {c}, the change in XOR is v_a ^ v_b ^ v_c = 0.
                    #   So XOR remains T.
                    #   Size change: -2 (remove a,b) + 1 (add c) = -1.
                    #   So this is good.
                    
                    sol_list = list(sol_set)
                    # Optimization: only consider a,b in sol_set
                    # For each a in sol_set:
                    #   For each b in sol_set (b>a):
                    #     target_mask = masks[a] ^ masks[b]
                    #     If target_mask in mask_to_indices:
                    #        For each c in mask_to_indices[target_mask]:
                    #           If c not in sol_set:
                    #              Remove a, b; Add c.
                    #              Break loops.
                    
                    found = False
                    for i in range(len(sol_list)):
                        if found: break
                        a = sol_list[i]
                        for j in range(i+1, len(sol_list)):
                            if found: break
                            b = sol_list[j]
                            tm = masks[a] ^ masks[b]
                            if tm in mask_to_indices:
                                for c in mask_to_indices[tm]:
                                    if c not in sol_set:
                                        sol_set.remove(a)
                                        sol_set.remove(b)
                                        sol_set.add(c)
                                        found = True
                                        break
                    if found:
                        # Retry pair reduction as well
                        pass # The while loop above already did pairs. This is a one-shot triple.
                        # We might need to iterate triple reduction too.
                        # Given time constraints, one pass might be enough.

                # Final check
                if len(sol_set) <= 100:
                    sol = list(sol_set)
                    prod = 210
                    for x in sol:
                        prod *= x
                    s = math.isqrt(prod)
                    if s * s == prod:
                        return [int(x) for x in sol]
        # Continue to next M

    # Fallback (should not be reached)
    return [840]

def _sa_refine(S, triples, pairs, n, t_start, deadline, cur_v, cur_size, best, best_size):
    """Simulated annealing refinement with 2-swap moves."""
    T = 1.0
    while time.time() < deadline:
        # 50% chance single flip, 50% chance 2-swap
        if random.random() < 0.5:
            v = random.randrange(n)
            old = S[v]
            if old == 0:
                addc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        addc += 1
                new_v = cur_v + addc
                new_size = cur_size + 1
            else:
                remc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        remc += 1
                new_v = cur_v - remc
                new_size = cur_size - 1
            
            new_score = new_size - 1000.0 * new_v
            cur_score = cur_size - 1000.0 * cur_v
            dE = new_score - cur_score
            
            if dE >= 0 or random.random() < math.exp(dE / T):
                S[v] = 1 - old
                cur_v = new_v
                cur_size = new_size
                if new_v == 0 and cur_size > best_size:
                    best[:] = S[:]
                    best_size = cur_size
        else:
            # 2-swap: pick one to remove, one to add
            # Pick random v_out from set
            candidates_out = [i for i in range(n) if S[i]]
            if not candidates_out:
                continue
            v_out = random.choice(candidates_out)
            
            # Pick random v_in from not in set
            candidates_in = [i for i in range(n) if not S[i]]
            if not candidates_in:
                continue
            v_in = random.choice(candidates_in)
            
            # Calculate change
            # Remove v_out, Add v_in
            # This is complex to calculate delta violations exactly without modifying S.
            # We can temporarily modify S, check, and revert if rejected.
            # Or calculate manually.
            
            # Manual calculation is safer for speed?
            # Let's just do the swap, check, and revert.
            S[v_out] = 0
            S[v_in] = 1
            
            # Count violations
            new_v = 0
            # We only need to check triples involving v_out or v_in
            # But checking all triples is O(N). N=180.
            # Number of triples is small.
            # Let's just count all violations for simplicity in this step, 
            # but optimize if too slow.
            # Actually, pairs list allows faster check.
            # Violation count is number of (i,j,k) in triples with S[i]&S[j]&S[k].
            
            # Efficient check:
            # A triple is violated if all 3 are in S.
            # We changed v_out (1->0) and v_in (0->1).
            # Triples involving v_out were potentially violated, now they are not (since v_out is 0).
            # Triples involving v_in were potentially not violated, now they might be.
            
            # Let's just recount for simplicity, as N is small.
            # Or use the pairs structure.
            # For each pair (u,w) in pairs[v_in], if S[u] and S[w], then (v_in, u, w) is a violation.
            # For each pair (u,w) in pairs[v_out], if S[u] and S[w], then (v_out, u, w) WAS a violation (before swap) but is NOT now.
            
            # We need to know how many violations existed BEFORE involving v_out.
            # And how many exist NOW involving v_in.
            
            # Let's compute addc and remc for the swap.
            # Note: S is already modified.
            # To find violations REMOVED by v_out=0:
            # We need to know if (v_out, u, w) was a violation BEFORE.
            # This requires knowing if S[u] and S[w] were 1 BEFORE.
            # But S is modified.
            # We can revert S temporarily to calculate?
            # Or we can store old state.
            
            # Let's do it simply:
            # 1. Save old S
            # 2. Apply swap
            # 3. Count new_v
            # 4. Compare
            # 5. Revert if rejected
            
            # This is O(N) per step.
            # N=180. Triples ~ 1000?
            # 1000 checks per step.
            # 1M steps in 1s? Maybe.
            
            # Let's try to be smart.
            # We will calculate `new_v` by iterating through `pairs[v_in]`.
            # And we need `old_v` contribution from `v_out`.
            # Let's just revert and recount if we want to be safe, 
            # but that's slow.
            
            # Better: 
            # Before swap:
            #   Violations involving v_out: count pairs (u,w) in pairs[v_out] where S[u] and S[w] are 1.
            #   Violations involving v_in: 0 (since v_in was 0).
            # After swap:
            #   Violations involving v_out: 0 (since v_out is 0).
            #   Violations involving v_in: count pairs (u,w) in pairs[v_in] where S[u] and S[w] are 1.
            
            # So:
            # removed_violations = count of (u,w) in pairs[v_out] with S[u] and S[w] (BEFORE swap)
            # added_violations = count of (u,w) in pairs[v_in] with S[u] and S[w] (AFTER swap)
            
            # We have S modified.
            # To get "BEFORE" state for v_out neighbors, we need to know if they were 1.
            # They are still 1 unless one of them was v_in.
            # If v_in is a neighbor of v_out, it was 0 before, now 1.
            # So if (v_out, v_in, w) is a triple, it was NOT a violation before (v_in=0), 
            # but IS a violation now (v_in=1, v_out=0? No, v_out=0 so it's NOT a violation).
            # Wait, if v_out=0, any triple with v_out is NOT a violation.
            # So violations involving v_out are ALWAYS removed.
            
            # So:
            # removed_violations = number of triples containing v_out that were valid violations BEFORE.
            # This is equal to number of pairs (u,w) in pairs[v_out] such that S[u] and S[w] were 1 BEFORE.
            # Since S is modified, S[u] and S[w] are 1 AFTER.
            # The only difference is if u or w is v_in.
            # If u == v_in, then S[u] was 0 BEFORE.
            # So we need to adjust.
            
            # Let's just do the simple recount for correctness first.
            new_v = 0
            for (i, j, k) in triples:
                if S[i] and S[j] and S[k]:
                    new_v += 1
            
            new_score = cur_size - 1000.0 * new_v # Size is same
            cur_score = cur_size - 1000.0 * cur_v
            dE = new_score - cur_score
            
            if dE >= 0 or random.random() < math.exp(dE / T):
                cur_v = new_v
                # cur_size unchanged
                if new_v == 0 and cur_size > best_size:
                    best[:] = S[:]
                    best_size = cur_size
            else:
                # Revert
                S[v_out] = 1
                S[v_in] = 0
        
        T *= 0.99999
        if T < 0.01:
            T = 0.01
    
    return S, cur_v, cur_size, best, best_size

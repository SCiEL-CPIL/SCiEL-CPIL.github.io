       def can_add_bit(v, M):
           # v is 1..128. Bit index is v-1.
           # Find U = { a in S | a+v in S }
           # a is in S -> bit a-1 in M is 1.
           # a+v is in S -> bit a+v-1 in M is 1.
           # So a is in U iff bit a-1 in M is 1 AND bit a+v-1 in M is 1.
           # This is equivalent to (M & (M >> v)) having bit a-1 set?
           # Let's check:
           # M has bit i set if i+1 in S.
           # M >> v has bit i set if M has bit i+v set. i.e., if (i+v)+1 in S -> i+v+1 in S.
           # We want a such that a in S (bit a-1) and a+v in S (bit a+v-1).
           # Let index be 0-based for values 0..127 corresponding to 1..128.
           # val = idx + 1.
           # Condition: (M & (1 << idx)) and (M & (1 << (idx+v))).
           # This is exactly (M & (M >> v)) ?
           # (M >> v) has bit idx set if M has bit idx+v set.
           # So (M & (M >> v)) has bit idx set if M has bit idx AND M has bit idx+v.
           # Yes.
           
           U_mask = M & (M >> v)
           
           # Now we need to check if there exist distinct a, b in U such that
           # (a+b in S) or (a+b == v).
           # Note: a, b are values 1..128.
           # If a+b == v, then a+b is not in S (since v not in S yet, and a,b < v usually? No, a,b can be anything).
           # Actually, if a+b == v, then the sum is v. Is v in S? No, we are adding it.
           # The condition is "a+b, a+c, b+c are also selected".
           # If we add v, the sums are a+v, b+v, a+b.
           # a+v and b+v are in S by definition of U.
           # So we just need a+b to be in S OR a+b == v?
           # Wait. If a+b == v, is that a problem?
           # The triple is (a,b,v). Sums: a+b, a+v, b+v.
           # If a+b == v, then the set of sums is {v, a+v, b+v}.
           # Are these "selected"?
           # v is being added. a+v and b+v are in S.
           # So yes, if a+b == v, then all three sums are in S (after adding v).
           # So we must forbid (a,b) in U if a+b == v OR a+b in S.
           
           # So for each pair (a,b) in U, a < b:
           # if (a+b == v) or (a+b in S): return False
           
           # How to do this fast?
           # Iterate through bits of U_mask.
           # If popcount(U_mask) is small, it's fast.
           # U_mask contains indices a-1.
           # Values are a = idx + 1.
           
           # Optimization: Precompute for each v, a bitset `sum_v_mask`? No.
           # We can precompute for each v, a list of pairs (a,b) such that a+b==v?
           # And for a+b in S, we need to check bit a+b-1 in M.
           
           # Let's just iterate.
           # To speed up, we can precompute `pairs_summing_to_v` for each v.
           # If any such pair is in U, conflict.
           # For a+b in S:
           # We need to check if there exist a,b in U such that bit a+b-1 is set in M.
           
           # This is still potentially O(k^2) where k is size of U.
           # But k is likely small for large v.
           
           # Better:
           # Precompute for each v, `conflict_pairs[v]` = list of (a,b) with a<b, a+b<=128.
           # This is too many.
           
           # Let's rely on the fact that for large v, U is small.
           # For small v, U is large, but we pick large v first?
           # The strategy "large_first_random" picks 128..65 first.
           # For v=128, a+128 in S implies a+128 <= 128 -> a=0 (not in S). So U is empty.
           # For v=127, a+127 in S -> a=1. U contains 1 if 1 in S and 128 in S.
           # So U is very small for large v.
           
           # So iterating U is efficient if we process vertices in descending order.
           
           # Let's implement `can_add_bit` with iteration over set bits of U_mask.
           
           # Get indices from U_mask.
           # While U_mask:
           #   lsb = U_mask & -U_mask
           #   idx = lsb.bit_length() - 1
           #   a = idx + 1
           #   ...
           
           # To check "a+b in S" or "a+b==v" for all pairs in U:
           # We can do this pairwise.
           
           # Even faster:
           # Precompute `sum_mask_for_pair`? No.
           
           # Let's just do the pairwise check on the small U.
           
           # One more optimization:
           # If we have U, we can check if (U & (U >> 1)) ... no.
           
           # Let's stick to the iterative check on U.
           
           # Extract values from U_mask.
           # Since U is small for large v, this is fine.
           
           # However, we can precompute `sums_to_v` bitset for each v?
           # For a fixed v, if a+b==v, then b = v-a.
           # So if a in U and v-a in U, and a != v-a, then conflict.
           # This can be checked with bit operations:
           # Let U_vals be the set of values.
           # We want to know if there exists a in U such that v-a in U and a != v/2.
           # This is equivalent to checking if U_mask has a bit at a-1 and at (v-a)-1.
           # (v-a)-1 = v - a - 1.
           # So we need to check if U_mask & (U_mask >> (v - 2 * a))? No.
           
           # Let's just use a simple loop for the "a+b==v" check and "a+b in S" check.
           # Given the small size of U for the greedy order (descending), this will be very fast.
           
           # Actually, we can precompute for each v, a bitset `partner_mask[v]`?
           # No, let's just write the efficient loop.
           
           # To make it even faster, we can precompute a list of bits set in U_mask?
           # No, extracting bits is fast.
           
           # Let's refine the "a+b in S" check.
           # For each a in U:
           #   For each b in U, b > a:
           #      if (a+b == v) or ((M >> (a+b-1)) & 1): return False
           
           # This is O(k^2).
           
           # Can we do better?
           # We want to know if there exists a,b in U such that a+b in S.
           # Let S_mask = M.
           # We want to check if (U * U) intersects S?
           # Not directly with bit ops.
           
           # But k is small.
           
           # Let's implement this.
           
           # Also, we must ensure a != b.
           
           # Implementation details:
           # M is int.
           # v is int 1..128.
           
           # 1. U_mask = M & (M >> v)
           # Note: M >> v shifts bits down.
           # Bit i in M corresponds to value i+1.
           # Bit i in (M >> v) corresponds to value i+v+1 in M?
           # Let's trace:
           # M has bit k set if value k+1 in S.
           # We want a such that value a in S (bit a-1) and value a+v in S (bit a+v-1).
           # (M >> v) has bit i set if M has bit i+v set.
           # So bit i in (M >> v) is set if value i+v+1 in S.
           # We want value a in S and value a+v in S.
           # Let i = a-1.
           # Bit i in M is set if a in S.
           # Bit i in (M >> v) is set if value i+v+1 = a-1+v+1 = a+v in S.
           # So yes, U_mask = M & (M >> v) correctly identifies indices a-1.
           
           # 2. Extract a's.
           # 3. Check pairs.
           
           # Precomputation for speed:
           # We can precompute `bits_set` for all possible masks? No, too many.
           
           # Let's just run it.
           
           # To improve restart count, we can run more trials.
           
           # Also, we can use a better heuristic:
           # Start with all numbers > 64.
           # Then add smaller ones.
           
           # The current code already does "large_first_random".
           # I will increase the number of trials because the check is now faster.
           
           # I will also add a local search step?
           # The directive says "without expensive local search".
           # So I will stick to greedy with many restarts.
           
           # I will increase trials to 5000 or 10000 if it's fast enough.
           # 128 vertices.
           # Each check is ~ O(k^2).
           # For v=128, k=0.
           # For v=100, k is small.
           # Average k is likely < 5.
           # So check is ~25 ops.
           # 128 * 25 = 3200 ops per trial.
           # 10000 trials = 32 million ops.
           # Python can do ~10-50 million simple ops/sec.
           # So 10000 trials should take < 1 sec.
           
           # Let's do 20000 trials to be safe and find better sets.
           
           # I will also try a few different random seeds or strategies.
           
           # One more optimization:
           # If we find a set of size 95+, we can stop early.
           
           # Let's write the code.

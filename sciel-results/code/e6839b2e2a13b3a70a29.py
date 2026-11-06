        sums = {0: [0]*15} # sum: coeffs
        # This is inefficient.
        # Better:
        # Generate all 3^15 vectors? No, too much memory for the vectors.
        # We just need to know if a sum exists.
        # So: set of sums? No, we need to reconstruct.
        # We can store: sum -> one representative index (0 to 3^15-1) or the coefficients themselves.
        # Storing coefficients: 14M * 15 ints. 210M ints. In Python, this is huge (~1.6GB). Too much.
        # Alternative: Store sum -> index. Then reconstruct from index.
        # 14M entries in a dict. Key: int (8 bytes), Value: int (8 bytes) + overhead. ~500MB? Maybe tight.
        # Alternative: Use a sorted list of sums and binary search.
        # But we need to map back to coefficients.
        #
        # Let's try a different split. 10 and 20?
        # 3^10 = 59,049. Small.
        # 3^20 = 3.4 billion. Too big.
        #
        # Let's stick to 15/15 but be smart.
        # We can generate the sums of the first half and store them in a `set` to check existence quickly?
        # If we find a sum, we need the coefficients.
        # We can regenerate the coefficients for the specific sum found? No, that's slow.
        #
        # Let's use a dictionary `sum_map` for the first half.
        # Key: sum value.
        # Value: The 15 coefficients as a tuple or a single integer encoding (base 3).
        # Encoding 15 digits in base 3: $3^{15} \approx 1.4 \times 10^7$. Fits in a 64-bit int.
        # So, `sum_map` maps `sum_val` -> `encoded_coeffs`.
        # Memory: 14M entries. Key (8 bytes) + Value (8 bytes) + Dict overhead (~50 bytes/entry).
        # 14M * 50 bytes = 700MB. This might exceed memory limits.
        #
        # Optimization: We don't need to store ALL sums if we are just looking for 1.
        # But we don't know which sums will pair with the second half.
        #
        # Alternative:
        # Sort the sums of the first half.
        # Sort the sums of the second half.
        # Use two pointers to find the pair with minimum absolute difference.
        # But we need to track the coefficients.
        #
        # Let's try to reduce the state space.
        # The weights $w_i = L/i$.
        # $L = 2329089562800$.
        # The sums are large integers.
        #
        # Maybe a simpler approach:
        # Use the current program's local search but start from a better initial guess?
        # The current program gets 200k.
        #
        # Let's look at the "Inspiration" scores.
        # Insp 3: 274.
        # Current: 200,683.
        #
        # If the seed is, say, $10^{10}$, then score 200k means $|S| \approx 10^{10}/2\cdot 10^5 = 50$.
        # If we get numerator 1, $|S| = 1/L \approx 4 \cdot 10^{-13}$.
        # Score would be $10^{10} / (4 \cdot 10^{-13}) = 2.5 \cdot 10^{22}$.
        # This is a massive improvement.
        #
        # So the goal is definitely to find numerator 1.
        #
        # How to find numerator 1 efficiently?
        #
        # We can use a BFS/DFS with pruning?
        # Or, we can exploit the structure.
        #
        # Let's try the MIM with 15/15 again, but optimize memory.
        # Instead of a dict, use a list of (sum, encoded) pairs, sort by sum.
        # 14M tuples of (int, int).
        # Python tuple overhead is high.
        #
        # Let's use NumPy.
        # Generate all 3^15 combinations of -1, 0, 1.
        # This is a 3^15 x 15 array.
        # 14M * 15 * 8 bytes (int64) = 1.7 GB. Too much.
        #
        # We can generate sums iteratively.
        # `sums = [0]`
        # `for w in weights[:15]:`
        #   `new_sums = []`
        #   `for s in sums:`
        #     `new_sums.append(s-w)`
        #     `new_sums.append(s)`
        #     `new_sums.append(s+w)`
        #   `sums = new_sums`
        #
        # This list will have 14M elements.
        # Python list of 14M ints: ~100MB.
        # We also need to map back to coefficients.
        #
        # We can store the coefficients in a parallel structure?
        # Or, we can just store the sums, find the best pair, and then *reconstruct* the coefficients for that specific sum using a second DFS?
        #
        # Plan:
        # 1. Generate all sums for first 15 weights. Store in a list `S1`.
        # 2. Sort `S1`.
        # 3. Generate all sums for last 15 weights. Store in a list `S2`.
        # 4. For each `s2` in `S2`:
        #    - Target `t = -s2`.
        #    - Binary search `S1` for `t`.
        #    - Check neighbors for `t+1` and `t-1` (i.e., sum = 1 or -1).
        #    - If found, we have the target sum `s1`.
        # 5. Once we have `s1` and `s2` such that `s1+s2 = 1`, we need to find the coefficients.
        #    - We run a DFS on the first 15 weights to find a combination that sums to `s1`.
        #    - We run a DFS on the last 15 weights to find a combination that sums to `s2`.
        #    - DFS on 15 items is fast ($3^{15}$ worst case, but with pruning it's much faster, or we can just iterate all 3^15 if we cached them? No, we didn't cache coefficients).
        #    - Actually, if we didn't cache coefficients, we have to search for them.
        #    - Searching for a specific sum in 15 items:
        #       The number of solutions might be large. We just need one.
        #       A simple DFS with pruning (suffix sums) should find one quickly.
        #
        # Memory for `S1` and `S2`:
        # 14M ints each. ~100MB each. Total 200MB. Acceptable.
        #
        # Time:
        # Generation: 14M operations. Fast.
        # Sorting: 14M log(14M) ~ 14M * 24 ~ 300M ops. NumPy sort is fast.
        # Search: 14M binary searches. 14M * 24 ~ 300M ops.
        # Total time: A few seconds.
        #
        # Let's implement this.


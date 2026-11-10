                # Try candidates from last_mark + 1 to limit
                # Use a sliding window of random candidates
                start = last_mark + 1
                # We'll try a few random candidates in a window
                # But we must ensure we eventually try all if needed?
                # No, if we don't find a solution in this branch, we backtrack.
                # So we can just pick random x's.
                
                # To be safe, let's just shuffle the list of candidates.
                # It's O(N log N) per node, which is fine for N=100.

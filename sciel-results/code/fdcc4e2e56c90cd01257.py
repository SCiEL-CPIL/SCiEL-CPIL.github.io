    import numpy as np

    def _count_progression_diffs(S):
        S_arr = np.array(S)
        S_set = set(S)
        diffs = set()
        # Vectorize pair differences
        # We need a, b, c in S such that b-a = c-b = d.
        # So 2b = a+c.
        # Iterate over all pairs (a, c) with a < c?
        # If a, c in S, and (a+c) is even, then b = (a+c)/2.
        # If b in S, then d = c-b = (c-a)/2 is a valid difference.
        # This reduces the check to:
        # For each pair (a, c) in S with a < c:
        #   if (a+c) % 2 == 0:
        #       b = (a+c)//2
        #       if b in S_set:
        #           diffs.add(c-b)
        # This is O(N^2) but simpler arithmetic.
        # N=32, N^2=1024.
        # Python loop is fine.
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a = S[i]
                c = S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in S_set:
                        diffs.add(c - b)
        return len(diffs)
    
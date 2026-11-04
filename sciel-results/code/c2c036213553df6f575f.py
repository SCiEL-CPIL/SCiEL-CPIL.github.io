   def greedy_construct():
       uncovered = (1 << n_triplets) - 1
       blocks = []
       while uncovered:
           # Pick a random uncovered triplet?
           # Current: lsb = uncovered & -uncovered (deterministic)
           # Change: Pick random set bit
           # ...
           # Current: Max gain
           # Change: Max gain with random tie-breaking or top-k sampling
   
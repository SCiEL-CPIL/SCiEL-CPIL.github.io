   def superperm(n):
       if n == 1:
           return ["1"]
       # Get superperm for n-1
       prev = superperm(n-1)
       # The length of prev[0] is L(n-1)
       # We need to construct n blocks
       # Actually, the structure is:
       # S_n is a concatenation of n blocks, each of length L(n-1) + (n-1)!
       # No, that's not right.
   
   def construct():
       candidates = []
       for i in range(1, 513):
           # check base 3
           n = i
           valid = True
           while n > 0:
               if n % 3 == 2:
                   valid = False
                   break
               n //= 3
           if valid:
               candidates.append(i)
       
       # Optional: try to add more
       # But 64 is a strong start.
       
       return candidates
   
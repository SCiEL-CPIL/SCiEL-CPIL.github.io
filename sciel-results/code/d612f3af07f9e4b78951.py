   def compute_max_book(adj, n):
       max_book = 0
       for u in range(n):
           for v in range(u+1, n):
               if (adj[u] >> v) & 1:
                   common = adj[u] & adj[v]
                   # Count bits
                   cnt = bin(common).count('1')
                   if cnt > max_book:
                       max_book = cnt
       return max_book
   
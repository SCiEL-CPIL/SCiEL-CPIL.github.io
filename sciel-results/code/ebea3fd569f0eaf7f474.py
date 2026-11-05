   def construct_4_10_cage():
       n = 320
       edges = []
       for i in range(n):
           for d in [1, 31]:
               j = (i + d) % n
               edges.append((i, j))
               j2 = (i - d) % n
               edges.append((i, j2))
       # Remove duplicates?
       # Since we add (i, i+d) and (i, i-d), and later (i+d, i) is not added directly but (i+d, i+d-d) = (i+d, i) IS added when loop is at i+d?
       # No, we iterate i from 0 to n-1.
       # For each i, we add edges to neighbors.
       # This will add each undirected edge twice: once from i to j, and once from j to i.
       # The problem says "list of edges (i, j)". Usually implies undirected, so (i,j) is same as (j,i).
       # "no repeated edges".
       # So I should add each edge only once.
       # If i < j, add (i, j).
       
       edges = set()
       for i in range(n):
           for d in [1, 31]:
               j = (i + d) % n
               if i < j:
                   edges.add((i, j))
               else:
                   edges.add((j, i))
       return n, list(edges)
   
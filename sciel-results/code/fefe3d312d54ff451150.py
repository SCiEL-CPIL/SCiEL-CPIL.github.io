import numpy as np
import time
import random

def construct():
      n = 40
      m = 401
      t_start = time.time()
      deadline = t_start + 110
      
      def get_score(adj):
          # adj is 40x40 numpy array of 0s and 1s
          max_book = 0
          # Iterate over all edges
          for i in range(n):
              for j in range(i+1, n):
                  if adj[i, j] == 1:
                      # Count common neighbors
                      # adj[i] & adj[j] gives vector of common neighbors
                      # Sum gives count
                      c = np.sum(adj[i] & adj[j])
                      if c > max_book:
                          max_book = c
          return max_book

      def random_graph():
          # Generate random graph with m edges
          edges = set()
          while len(edges) < m:
              u = random.randint(0, n-1)
              v = random.randint(0, n-1)
              if u != v:
                  if u > v: u, v = v, u
                  edges.add((u, v))
          adj = np.zeros((n, n), dtype=np.uint8)
          for u, v in edges:
              adj[u, v] = 1
              adj[v, u] = 1
          return adj, list(edges)

      best_score = 100
      best_edges = None
      best_adj = None

      # Initial random restarts to find a good start?
      # Or just one long run.
      
      current_adj, current_edges = random_graph()
      current_score = get_score(current_adj)
      
      if current_score < best_score:
          best_score = current_score
          best_edges = current_edges[:]
          best_adj = current_adj.copy()
          if best_score == 7:
              return [[u, v] for u, v in best_edges]

      steps = 0
      last_improve = time.time()
      
      while time.time() < deadline:
          # Check for stagnation
          if time.time() - last_improve > 5.0:
              # Restart
              current_adj, current_edges = random_graph()
              current_score = get_score(current_adj)
              last_improve = time.time()
              continue

          # Pick random edge to remove
          out_u, out_v = random.choice(current_edges)
          
          # Pick random non-edge to add
          while True:
              in_u = random.randint(0, n-1)
              in_v = random.randint(0, n-1)
              if in_u == in_v: continue
              if in_u > in_v: in_u, in_v = in_v, in_u
              # Check if it's the same as out edge
              if (in_u, in_v) == (out_u, out_v): continue
              # Check if it's already in graph
              if current_adj[in_u, in_v] == 0:
                  break

          # Apply swap
          current_adj[out_u, out_v] = 0
          current_adj[out_v, out_u] = 0
          current_edges.remove((out_u, out_v))
          
          current_adj[in_u, in_v] = 1
          current_adj[in_v, in_u] = 1
          current_edges.append((in_u, in_v))

          new_score = get_score(current_adj)

          if new_score < current_score:
              current_score = new_score
              last_improve = time.time()
              if current_score < best_score:
                  best_score = current_score
                  best_edges = current_edges[:]
                  best_adj = current_adj.copy()
                  if best_score == 7:
                      return [[u, v] for u, v in best_edges]
          else:
              # Revert
              current_adj[in_u, in_v] = 0
              current_adj[in_v, in_u] = 0
              current_edges.remove((in_u, in_v))
              
              current_adj[out_u, out_v] = 1
              current_adj[out_v, out_u] = 1
              current_edges.append((out_u, out_v))

      # Return best found
      return [[u, v] for u, v in best_edges]

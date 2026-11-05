import time
from collections import defaultdict

def construct():
       start_time = time.time()
       deadline = start_time + 60  # Safety margin for 90s limit
       
       # 1. Run Greedy to get a good initial lower bound
       def run_greedy():
           values, products = [], set()
           for x in range(2, 129):
               tails, prod = [x], x
               for a in reversed(values):
                   prod *= a
                   tails.append(prod)
               if not products.intersection(tails):
                   values.append(x)
                   products.update(tails)
           return values, products
       
       best_seq, best_prods = run_greedy()
       best_len = len(best_seq)
       
       # 2. Beam Search
       # State: (seq_tuple, prods_set)
       # We start with the empty state? Or the greedy state?
       # Beam search usually starts from the root.
       
       beam = [((), frozenset())]
       
       for x in range(2, 129):
           if time.time() > deadline:
               break
               
           next_beam = []
           for seq, prods in beam:
               # Option 1: Skip x
               next_beam.append((seq, prods))
               
               # Option 2: Include x
               if seq and seq[-1] >= x: # Should not happen if we iterate x
                   continue
               
               # Compute tails
               tails = [x]
               prod = x
               for a in reversed(seq):
                   prod *= a
                   tails.append(prod)
                   
               # Check collision
               if not any(t in prods for t in tails):
                   new_prods = prods.union(tails)
                   next_beam.append((seq + (x,), new_prods))
           
           # Prune beam
           # Sort by length descending, then by "potential"
           # Potential heuristic: smaller max product? Or just length.
           # If lengths are equal, we might prefer sequences that used smaller numbers?
           # Or sequences with more "room"?
           
           # To avoid duplicates in the beam (same seq), we can use a set of seqs?
           # But different prods sets for same seq is impossible (prods are determined by seq).
           # So we can just keep unique seqs.
           
           unique_states = {}
           for seq, prods in next_beam:
               if seq not in unique_states:
                   unique_states[seq] = prods
                   
           # Sort keys by length desc, then by sum of elements (smaller is better?)
           # Or by the last element (smaller is better?)
           sorted_seqs = sorted(unique_states.keys(), key=lambda s: (-len(s), s[-1] if s else 0))
           
           # Keep top W
           W = 100
           beam = [(s, unique_states[s]) for s in sorted_seqs[:W]]
           
           # Update best
           if len(sorted_seqs[0]) > best_len:
               best_len = len(sorted_seqs[0])
               best_seq = list(sorted_seqs[0])
               best_prods = unique_states[sorted_seqs[0]]
               
       return list(best_seq)

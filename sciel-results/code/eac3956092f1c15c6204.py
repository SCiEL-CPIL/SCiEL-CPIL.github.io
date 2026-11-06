   import math
   import numpy as np
   from scipy.optimize import minimize
   
   def _jc_likelihood(tree, seqs):
       # tree: dict {node: (parent, [children])} or similar
       # Need a good tree representation.
       pass
   
   def _optimize(tree, seqs):
       pass
   
   def reconstruct(sequences):
       # 1. Build NJ Tree
       # 2. Convert to internal representation
       # 3. Optimize Branch Lengths
       # 4. NNI Polish
       # 5. Convert to Newick
       pass
   
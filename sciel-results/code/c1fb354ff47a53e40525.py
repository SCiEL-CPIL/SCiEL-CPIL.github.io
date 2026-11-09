import time
import sys

# Increase recursion limit just in case, though depth is < 128
sys.setrecursionlimit(20000)

class Solver:
    def __init__(self):
        self.best_len = 0
        self.best_seq = []
        self.deadline = time.time() + 55.0  # 55s safety margin
        self.timeout = False
        self.nodes_explored = 0
        self.global_products = set()

    def check_conflict(self, suffix_prods, x):
        """
        Check if adding x to seq creates any duplicate product.
        Uses self.global_products for collision check.
        Returns: (bool, new_suffix_prods, new_products_list)
        """
        new_prods = []
        prod = x
        new_prods.append(prod)
        if prod in self.global_products:
            return False, None, None
        
        # Multiply by elements of seq in reverse order (via suffix_prods)
        # suffix_prods[i] is product of seq[i:]
        # We need x * seq[-1], x * seq[-1] * seq[-2], etc.
        # This corresponds to x * suffix_prods[-1], x * suffix_prods[-2], ...
        
        for i in range(len(suffix_prods) - 1, -1, -1):
            prod *= suffix_prods[i]
            new_prods.append(prod)
            if prod in self.global_products:
                return False, None, None
        
        # new_prods is [x, x*last, ..., x*whole]
        # We want new_suffix_prods in [whole, ..., last] format to match input convention
        new_suffix = list(reversed(new_prods))
        
        return True, new_suffix, new_prods

    def search(self, seq, suffix_prods):
        if self.timeout:
            return
        
        self.nodes_explored += 1
        if self.nodes_explored % 1000 == 0:
            if time.time() > self.deadline:
                self.timeout = True
                return

        current_len = len(seq)
        
        # Update best
        if current_len > self.best_len:
            self.best_len = current_len
            self.best_seq = seq.copy()
        
        # Pruning: Can we possibly beat the best?
        if seq:
            max_possible = 128 - seq[-1]
        else:
            max_possible = 127  # 2 to 128
            
        if current_len + max_possible <= self.best_len:
            return

        # Find candidates
        start_val = seq[-1] + 1 if seq else 2
        end_val = 129
        
        for x in range(start_val, end_val):
            if self.timeout:
                return
            
            valid, new_suffix, new_prods = self.check_conflict(suffix_prods, x)
            if valid:
                # Add new products to global set
                for p in new_prods:
                    self.global_products.add(p)
                
                # Recurse
                seq.append(x)
                self.search(seq, new_suffix)
                seq.pop()
                
                # Backtrack: Remove new products from global set
                for p in new_prods:
                    self.global_products.remove(p)

def construct():
    solver = Solver()
    
    # Initial greedy to set a good lower bound
    greedy_seq = []
    greedy_suffix = []
    greedy_prods = set()
    
    for x in range(2, 129):
        new_prods = []
        prod = x
        new_prods.append(prod)
        if prod in greedy_prods:
            continue
        
        valid = True
        for i in range(len(greedy_suffix) - 1, -1, -1):
            prod *= greedy_suffix[i]
            new_prods.append(prod)
            if prod in greedy_prods:
                valid = False
                break
        
        if valid:
            greedy_seq.append(x)
            new_suffix = list(reversed(new_prods))
            greedy_suffix = new_suffix
            greedy_prods.update(new_prods)
            
    if len(greedy_seq) > solver.best_len:
        solver.best_len = len(greedy_seq)
        solver.best_seq = greedy_seq.copy()

    # Start search from empty sequence
    solver.search([], [])
    
    return solver.best_seq

"""
Structural Change: Symbolic Regression via Genetic Programming.
Instead of approximating the function with a non-parametric model (like GP) or a fixed basis expansion,
this strategy explicitly searches for a closed-form mathematical formula using Genetic Programming (GP).

Key Structural Differences:
1. **Representation**: The model is a symbolic expression tree (e.g., `sin(x1) * x2 + 1`), not a vector of weights or a kernel matrix.
2. **Search Method**: Evolutionary search (crossover, mutation, selection) over the space of expressions, rather than gradient-based or grid search over continuous hyperparameters.
3. **Generalization**: By finding the exact underlying formula (or a close approximation), it generalizes perfectly to test points without overfitting to noise (though data is noiseless here, the parsimony pressure prevents overfitting to random fluctuations if any).
4. **Efficiency**: GP is computationally expensive per evaluation but can find the exact solution with fewer parameters than a GP interpolant, leading to faster prediction and robustness against the "mean over equations" scoring where one bad fit ruins the score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
import random
import math

# Set random seed for reproducibility in case of multiple runs within the same process if needed, 
# though the evaluator likely resets state. We use a fixed seed for the GP internal randomness 
# to ensure consistent behavior if called multiple times, but here we assume one call per equation.
# Actually, better to let it be random or seed based on input hash for diversity.
# We will seed based on the first element of X to be deterministic per input but diverse across inputs.

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape

    # Normalize inputs to [0, 1] to help GP convergence and prevent overflow in exp/log
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # Define the set of primitives (functions and terminals)
    # Terminals: x1, x2, x3, ... and constants
    # Functions: +, -, *, /, sin, cos, exp, log, abs, sqrt, tan
    
    # We limit the number of terminals to the actual number of features
    # and a small set of constants.
    
    class GPNode:
        def __init__(self, func=None, children=None, terminal=None):
            self.func = func
            self.children = children
            self.terminal = terminal
            
        def __repr__(self):
            if self.terminal is not None:
                return str(self.terminal)
            if self.func is not None:
                if len(self.children) == 0:
                    return self.func.__name__
                elif len(self.children) == 1:
                    return f"{self.func.__name__}({self.children[0]})"
                else:
                    return f"{self.func.__name__}({', '.join([str(c) for c in self.children])})"
            return "Node"

    # Define safe functions
    def safe_div(a, b):
        # Avoid division by zero
        return np.where(np.abs(b) < 1e-6, 0, a / b)

    def safe_log(a):
        return np.log(np.maximum(a, 1e-6))

    def safe_sqrt(a):
        return np.sqrt(np.maximum(a, 0))

    def safe_exp(a):
        # Clip to prevent overflow
        return np.exp(np.clip(a, -50, 50))

    def safe_tan(a):
        return np.tan(np.clip(a, -np.pi/2 + 0.1, np.pi/2 - 0.1))

    # Function set
    # arity 2
    func2 = [
        (lambda a, b: a + b, 2, "add"),
        (lambda a, b: a - b, 2, "sub"),
        (lambda a, b: a * b, 2, "mul"),
        (lambda a, b: safe_div(a, b), 2, "div"),
    ]
    # arity 1
    func1 = [
        (lambda a: np.sin(a), 1, "sin"),
        (lambda a: np.cos(a), 1, "cos"),
        (lambda a: safe_exp(a), 1, "exp"),
        (lambda a: safe_log(a), 1, "log"),
        (lambda a: np.abs(a), 1, "abs"),
        (lambda a: safe_sqrt(a), 1, "sqrt"),
        # (lambda a: safe_tan(a), 1, "tan"), # Tan can be unstable
    ]
    
    all_funcs = func2 + func1
    
    # Terminals: x1...xn, and constants
    # We use indices 0..n_features-1 for x, and specific floats for constants
    # To keep the tree size manageable, we limit constants to a few values
    constants = [0.5, 1.0, 2.0, 0.1, 10.0]
    
    # GP Parameters
    pop_size = 200
    max_depth = 8
    max_generations = 300
    tournament_size = 5
    crossover_rate = 0.7
    mutation_rate = 0.3
    
    # Seed the random number generator based on input to ensure reproducibility per input
    seed_val = int(abs(X[0, 0] * 1000)) % (2**32)
    rng = np.random.RandomState(seed_val)
    
    def random_tree(max_depth, min_depth=1):
        """Generate a random expression tree."""
        depth = rng.randint(min_depth, max_depth + 1)
        return _random_tree(depth)
    
    def _random_tree(depth):
        if depth <= 0 or (depth == 1 and rng.rand() < 0.5):
            # Terminal
            if rng.rand() < 0.8:
                # Feature
                idx = rng.randint(0, n_features)
                return GPNode(terminal=idx)
            else:
                # Constant
                const = constants[rng.randint(0, len(constants))]
                return GPNode(terminal=const)
        else:
            # Function
            func_tuple = all_funcs[rng.randint(0, len(all_funcs))]
            arity = func_tuple[1]
            children = []
            for _ in range(arity):
                children.append(_random_tree(depth - 1))
            return GPNode(func=func_tuple[0], children=children)
    
    def evaluate_tree(node, X_data):
        """Evaluate the expression tree on the dataset."""
        if node.terminal is not None:
            if isinstance(node.terminal, int):
                return X_data[:, node.terminal]
            else:
                return np.full(len(X_data), node.terminal)
        else:
            if len(node.children) == 1:
                child_val = evaluate_tree(node.children[0], X_data)
                return node.func(child_val)
            else:
                c1 = evaluate_tree(node.children[0], X_data)
                c2 = evaluate_tree(node.children[1], X_data)
                return node.func(c1, c2)
    
    def tree_depth(node):
        if node.terminal is not None:
            return 0
        max_child_depth = 0
        for c in node.children:
            max_child_depth = max(max_child_depth, tree_depth(c))
        return max_child_depth + 1
    
    def tree_size(node):
        if node.terminal is not None:
            return 1
        size = 1
        for c in node.children:
            size += tree_size(c)
        return size
    
    def get_fitness(node):
        """Calculate MSE + Parsimony penalty."""
        try:
            pred = evaluate_tree(node, X_norm)
            mse = np.mean((pred - y) ** 2)
            # Parsimony penalty to prefer simpler expressions
            size = tree_size(node)
            penalty = 0.001 * size
            return mse + penalty
        except Exception:
            return float('inf')
    
    def crossover(node1, node2):
        """Standard subtree crossover."""
        if node1.terminal is not None or node2.terminal is not None:
            return node1, node2
        
        # Pick a random non-terminal node in node1
        def pick_subtree(node):
            # List all non-terminal nodes
            candidates = []
            def _collect(n):
                if n.terminal is not None:
                    return
                candidates.append(n)
                for c in n.children:
                    _collect(c)
            _collect(node)
            if not candidates:
                return node
            return candidates[rng.randint(0, len(candidates))]
        
        sub1 = pick_subtree(node1)
        sub2 = pick_subtree(node2)
        
        # Swap subtrees
        # We need to replace sub1 in node1 with sub2, and sub2 in node2 with sub1
        # This requires careful pointer manipulation or copying.
        # Since Python objects are references, we can't easily swap without rebuilding.
        # Simpler approach: Deep copy and rebuild? Or just return the subtrees and let the caller handle?
        # For simplicity and safety, we'll do a "swap" by creating new nodes that reference the other subtree.
        # Actually, standard GP swaps the subtrees.
        
        # Let's implement a simple swap by finding the parent and index.
        # This is complex. Alternative: "Point Mutation" and "Subtree Replacement".
        # Let's stick to a simpler mutation: replace a random subtree with a new random one.
        # And Crossover: take a random subtree from parent1 and attach to parent2? No, that changes arity.
        
        # Let's use a simpler Crossover:
        # 1. Pick a point in Parent 1 (a subtree)
        # 2. Pick a point in Parent 2 (a subtree)
        # 3. Swap them.
        
        # To make swapping easy, we'll use a recursive copy approach or just return the new trees.
        # Given the small size, deep copy is acceptable.
        
        import copy
        n1_copy = copy.deepcopy(node1)
        n2_copy = copy.deepcopy(node2)
        
        # Replace sub1 in n1_copy with sub2 (deep copy of sub2)
        # And sub2 in n2_copy with sub1 (deep copy of sub1)
        
        def replace_subtree(node, target_subtree, new_subtree):
            if node is target_subtree:
                return copy.deepcopy(new_subtree)
            if node.terminal is not None:
                return node
            for i, child in enumerate(node.children):
                node.children[i] = replace_subtree(child, target_subtree, new_subtree)
            return node
        
        # Note: replace_subtree modifies the tree in place. 
        # We need to be careful that sub1 and sub2 are distinct objects from the original parents.
        # Since we deep copied n1 and n2, sub1 (from n1) and sub2 (from n2) are now in n1_copy and n2_copy?
        # No, sub1 was picked from node1 (original). We need to pick from the copies.
        
        sub1_in_copy = pick_subtree(n1_copy)
        sub2_in_copy = pick_subtree(n2_copy)
        
        # Now swap
        # This is tricky because replace_subtree modifies the tree.
        # Let's just return the original nodes if crossover is too complex/risky for time.
        # Actually, let's just do Mutation for simplicity and robustness.
        # GP can work with high mutation rates.
        return node1, node2 # Fallback: no crossover, just return parents
        
    def mutate(node):
        """Randomly mutate a node."""
        if rng.rand() > mutation_rate:
            return node
        
        import copy
        new_node = copy.deepcopy(node)
        
        # Pick a random node in the tree to mutate
        def pick_any_node(node):
            candidates = []
            def _collect(n):
                candidates.append(n)
                if n.terminal is not None:
                    return
                for c in n.children:
                    _collect(c)
            _collect(node)
            return candidates[rng.randint(0, len(candidates))]
        
        target = pick_any_node(new_node)
        
        if target.terminal is not None:
            # Mutate terminal
            if rng.rand() < 0.5:
                # Change feature
                target.terminal = rng.randint(0, n_features)
            else:
                # Change constant
                target.terminal = constants[rng.randint(0, len(constants))]
        else:
            # Mutate function
            # Replace function with a random function of same arity? Or different?
            # If different arity, we must regenerate children.
            # For simplicity, replace with same arity function.
            arity = len(target.children)
            if arity == 1:
                new_func = func1[rng.randint(0, len(func1))][0]
                target.func = new_func
            else:
                new_func = func2[rng.randint(0, len(func2))][0]
                target.func = new_func
            
            # Optionally, replace a child with a new random subtree
            if rng.rand() < 0.5 and target.children:
                idx = rng.randint(0, len(target.children) - 1)
                target.children[idx] = random_tree(max_depth=4)
                
        return new_node

    # Initialize Population
    population = []
    for _ in range(pop_size):
        pop = random_tree(max_depth)
        population.append(pop)
    
    # Evaluate initial population
    fitnesses = [get_fitness(ind) for ind in population]
    
    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    # Evolution Loop
    for gen in range(max_generations):
        if time.time() > deadline - 10:
            break
            
        # Selection: Tournament
        next_generation = []
        
        # Keep the best individual (Elitism)
        next_generation.append(best_individual)
        
        while len(next_generation) < pop_size:
            # Pick 2 parents via tournament
            parent1_idx = _tournament_select(fitnesses, population)
            parent2_idx = _tournament_select(fitnesses, population)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover (currently disabled/fallback to parents)
            if rng.rand() < crossover_rate:
                # Simple crossover: swap random subtrees
                # Implementing a robust swap is complex. 
                # Let's try a simple "Subtree Replacement" crossover:
                # Take a subtree from P1 and replace a random subtree in P2 with it.
                # And vice versa? Or just one way?
                # Let's do: Child1 = P1 with a random subtree replaced by a random subtree from P2.
                # Child2 = P2 with a random subtree replaced by a random subtree from P1.
                
                import copy
                c1 = copy.deepcopy(parent1)
                c2 = copy.deepcopy(parent2)
                
                # Pick a subtree from P2 to inject into C1
                sub_from_p2 = _pick_random_subtree(parent2)
                # Pick a location in C1 to replace
                loc_in_c1 = _pick_random_subtree(c1)
                
                # Replace loc_in_c1 with sub_from_p2
                c1 = _replace_subtree(c1, loc_in_c1, sub_from_p2)
                
                # Pick a subtree from P1 to inject into C2
                sub_from_p1 = _pick_random_subtree(parent1)
                loc_in_c2 = _pick_random_subtree(c2)
                
                c2 = _replace_subtree(c2, loc_in_c2, sub_from_p1)
                
                # Mutate
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                next_generation.extend([c1, c2])
            else:
                # Mutation only
                c1 = mutate(parent1)
                c2 = mutate(parent2)
                next_generation.extend([c1, c2])
        
        # Evaluate new population
        population = next_generation[:pop_size]
        fitnesses = [get_fitness(ind) for ind in population]
        
        # Update best
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx]
            
        # Early stopping if fitness is very low
        if best_fitness < 1e-6:
            break

    # Final Prediction
    # We use the best individual found
    final_tree = best_individual
    
    # If GP failed to find a good fit, fallback to a simple linear model or polynomial
    # Check if best_fitness is reasonable
    if best_fitness > 1.0: # High error
        # Fallback to Polynomial Regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            return features(X) @ w
        return predict

    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals) / range_vals
        try:
            pred = evaluate_tree(final_tree, X_norm_test)
            return pred
        except Exception:
            # Fallback to mean if prediction fails
            return np.full(len(X), np.mean(y_train))
    
    return predict

# Helper functions for GP operations

def _tournament_select(fitnesses, population, k=5):
    indices = np.random.choice(len(fitnesses), k, replace=False)
    min_idx = min(indices, key=lambda i: fitnesses[i])
    return min_idx

def _pick_random_subtree(node):
    """Pick a random subtree from the tree."""
    candidates = []
    def _collect(n):
        candidates.append(n)
        if n.terminal is not None:
            return
        for c in n.children:
            _collect(c)
    _collect(node)
    return candidates[np.random.randint(0, len(candidates))]

def _replace_subtree(node, target, new_subtree):
    """Replace the subtree at 'target' with 'new_subtree' in the tree rooted at 'node'."""
    import copy
    if node is target:
        return copy.deepcopy(new_subtree)
    if node.terminal is not None:
        return node
    for i, child in enumerate(node.children):
        node.children[i] = _replace_subtree(child, target, new_subtree)
    return node

# EVOLVE-BLOCK-END

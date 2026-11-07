# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_FIT_DEADLINE = 260.0  # safety margin under 300s
_START = None

# Symbol set
def _mk_func_set():
    funcs = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: np.where(np.abs(b) < 1e-8, 1e6, a / np.where(np.abs(b) < 1e-8, 1e-8, b)),
        'neg': lambda a: -a,
        'abs': lambda a: np.abs(a),
        'sqrt': lambda a: np.sqrt(np.maximum(a, 0)),
        'exp': lambda a: np.exp(np.clip(a, -30, 30)),
        'log': lambda a: np.log(np.maximum(np.abs(a), 1e-8)),
        'sin': lambda a: np.sin(a),
        'cos': lambda a: np.cos(a),
        'tan': lambda a: np.tan(np.clip(a, -1.4, 1.4)),
        'tanh': lambda a: np.tanh(a),
        'pow2': lambda a: a * a,
        'pow3': lambda a: a * a * a,
    }
    return funcs

FUNC_SET = _mk_func_set()
FUNC_NAMES = list(FUNC_SET.keys())
UNARY = ['neg', 'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'tanh', 'pow2', 'pow3']
BINARY = ['add', 'sub', 'mul', 'div']
MAX_DEPTH = 6

def _new_var(rng, n_vars):
    return ('var', rng.randint(0, n_vars - 1))

def _new_const(rng):
    return ('const', float(rng.uniform(-3, 3)))

def _new_node(rng, n_vars, depth=0, max_depth=MAX_DEPTH):
    if depth >= max_depth:
        return rng.choice([_new_var(rng, n_vars), _new_const(rng)])
    r = rng.random()
    if r < 0.55:
        name = rng.choice(BINARY)
        a = _new_node(rng, n_vars, depth + 1, max_depth)
        b = _new_node(rng, n_vars, depth + 1, max_depth)
        return (name, a, b)
    elif r < 0.95:
        name = rng.choice(UNARY)
        a = _new_node(rng, n_vars, depth + 1, max_depth)
        return (name, a)
    else:
        return _new_var(rng, n_vars)

def _depth(node):
    if isinstance(node, tuple):
        if len(node) == 2:
            return 1 + _depth(node[1])
        return 1 + max(_depth(node[1]), _depth(node[2]))
    return 0

def _size(node):
    if isinstance(node, tuple):
        if len(node) == 2:
            return 1 + _size(node[1])
        return 1 + _size(node[1]) + _size(node[2])
    return 1

def _eval_node(node, X):
    """X: (n, d). Returns (n,) array."""
    if isinstance(node, tuple):
        name = node[0]
        if name == 'var':
            return X[:, node[1]]
        if name == 'const':
            return np.full(len(X), node[1])
        if len(node) == 2:
            a = _eval_node(node[1], X)
            return FUNC_SET[name](a)
        else:
            a = _eval_node(node[1], X)
            b = _eval_node(node[2], X)
            return FUNC_SET[name](a, b)
    raise ValueError

def _safe_eval(node, X):
    try:
        y = _eval_node(node, X)
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def _nse(node, X, y, var_y):
    yhat = _safe_eval(node, X)
    if yhat is None:
        return float('inf')
    err = np.mean((yhat - y) ** 2)
    return err / max(var_y, 1e-12)

def _crossover(rng, a, b, n_vars):
    # subtree crossover
    if _size(a) < 2 or _size(b) < 2:
        return _new_node(rng, n_vars)
    # pick random node in a to replace with random subtree from b
    def pick(node, depth=0):
        if not isinstance(node, tuple) or _size(node) < 2:
            return node
        r = rng.random()
        if len(node) == 2:
            if r < 0.5:
                return (node[0], pick(node[1], depth + 1))
            return node
        else:
            if r < 0.5:
                return (node[0], pick(node[1], depth + 1), node[2])
            elif r < 0.9:
                return (node[0], node[1], pick(node[2], depth + 1))
            return node
    new_a = pick(a)
    return new_a

def _mutate(rng, node, n_vars):
    if _size(node) < 2:
        return _new_node(rng, n_vars)
    def rec(node):
        if not isinstance(node, tuple):
            return node
        r = rng.random()
        if r < 0.15:
            return _new_node(rng, n_vars)
        if len(node) == 2:
            return (node[0], rec(node[1]))
        else:
            if r < 0.5:
                return (node[0], rec(node[1]), node[2])
            elif r < 0.8:
                return (node[0], node[1], rec(node[2]))
            return node
    return rec(node)

def _poly_predict(X, y, degree=3):
    n, d = X.shape
    # Build polynomial features up to degree for d=1 (or low degree for d>1)
    if d == 1:
        x = X[:, 0]
        cols = [np.ones(n)]
        for k in range(1, degree + 1):
            cols.append(x ** k)
        F = np.column_stack(cols)
    else:
        # low-degree monomials
        cols = [np.ones(n)]
        for k in range(1, degree + 1):
            # all monomials of total degree k
            for i in range(d):
                for j in range(i, d):
                    if k == 1:
                        if i == j:
                            cols.append(X[:, i])
                    else:
                        # only a few to keep small
                        pass
        # simpler: just linear + squares
        cols = [np.ones(n)]
        for i in range(d):
            cols.append(X[:, i])
            cols.append(X[:, i] ** 2)
        F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    def predict(Xt):
        n2 = Xt.shape[0]
        if d == 1:
            x = Xt[:, 0]
            cols = [np.ones(n2)]
            for k in range(1, degree + 1):
                cols.append(x ** k)
            F2 = np.column_stack(cols)
        else:
            cols = [np.ones(n2)]
            for i in range(d):
                cols.append(Xt[:, i])
                cols.append(Xt[:, i] ** 2)
            F2 = np.column_stack(cols)
        return F2 @ w
    return predict

def fit(X_train, y_train):
    global _START
    if _START is None:
        _START = time.time()
    rng = random.Random(42)
    np.random.seed(42)

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    var_y = float(np.var(y))
    if var_y < 1e-12:
        # constant
        c = float(np.mean(y))
        return lambda X: np.full(len(X), c)

    # Initialize population
    pop_size = 60
    population = []
    # seed with some simple structures
    for i in range(pop_size):
        node = _new_node(rng, d, depth=0, max_depth=MAX_DEPTH)
        population.append(node)

    # Evaluate initial
    scores = []
    for node in population:
        scores.append(_nse(node, X, y, var_y))

    best_node = population[0]
    best_score = scores[0]
    for i in range(len(population)):
        if scores[i] < best_score:
            best_score = scores[i]
            best_node = population[i]

    # GP loop with time guard
    while time.time() - _START < _FIT_DEADLINE:
        # Elitism: keep top 10%
        order = np.argsort(scores)
        elite_count = max(5, pop_size // 10)
        elite = [population[i] for i in order[:elite_count]]
        elite_scores = [scores[i] for i in order[:elite_count]]

        new_pop = elite[:]
        new_scores = elite_scores[:]

        # Fill rest with offspring
        for _ in range(pop_size - len(new_pop)):
            # tournament selection
            def select():
                idxs = rng.sample(range(len(population)), min(3, len(population)))
                best_i = min(idxs, key=lambda i: scores[i])
                return population[best_i]

            a = select()
            b = select()
            r = rng.random()
            if r < 0.5:
                child = _crossover(rng, a, b, d)
            elif r < 0.8:
                child = _mutate(rng, a, d)
            else:
                child = _new_node(rng, d)
            # limit size
            if _size(child) > 40:
                child = _new_node(rng, d)
            s = _nse(child, X, y, var_y)
            new_pop.append(child)
            new_scores.append(s)

        population = new_pop
        scores = new_scores

        for i in range(len(population)):
            if scores[i] < best_score:
                best_score = scores[i]
                best_node = population[i]

        # If very good, stop early
        if best_score < 1e-4:
            break

    # Final prediction
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        yhat = _safe_eval(best_node, Xt)
        if yhat is not None and np.all(np.isfinite(yhat)):
            return yhat
        # fallback to polynomial
        return _poly_predict(X, y, degree=3)(Xt)

    return predict
# EVOLVE-BLOCK-END

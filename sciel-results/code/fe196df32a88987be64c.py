"""Symbolic regression via evolutionary search + Gaussian Process baseline."""
import numpy as np
import time
import random
import operator
import math

# ---------- Operator / terminal tables (1D, x is the single input) ----------
def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mul(a, b): return a * b
def _safe_div(a, b): return a / np.where(np.abs(b) < 1e-8, 1e-8, b)
def _sin(a, b): return np.sin(a)
def _cos(a, b): return np.cos(a)
def _tan(a, b): return np.tanh(a)
def _exp(a, b): return np.exp(np.clip(a, -50, 50))
def _sqrt(a, b): return np.sqrt(np.abs(a))
def _abs(a, b): return np.abs(a)
def _log(a, b): return np.log(np.abs(a) + 1e-8)
def _x(a, b): return a
def _pi(a, b): return np.pi
def _e(a, b): return np.e
def _one(a, b): return 1.0
def _two(a, b): return 2.0
def _three(a, b): return 3.0
def _neg(a, b): return -a

# arity-2 functions: name -> (func, arity)
BIN_FUNCS = {
    'add': _add, 'sub': _sub, 'mul': _mul, 'div': _safe_div,
    'sin': _sin, 'cos': _cos, 'tan': _tan, 'exp': _exp,
    'sqrt': _sqrt, 'abs': _abs, 'log': _log, 'neg': _neg,
}
# arity-1 "functions" (unary)
UNARY_FUNCS = {'sin': _sin, 'cos': _cos, 'tan': _tan, 'exp': _exp,
               'sqrt': _sqrt, 'abs': _abs, 'log': _log, 'neg': _neg}
# terminals
TERMINALS = ['x', 'pi', 'e', '1', '2', '3']
TERM_FUNCS = {'x': _x, 'pi': _pi, 'e': _e, '1': _one, '2': _two, '3': _three}

# ---------- Expression tree (list of (name, left, right)) ----------
def _make_leaf():
    name = random.choice(TERMINALS)
    return (name, None, None)

def _make_node():
    if random.random() < 0.5:
        # unary applied to a leaf or small subtree
        fn = random.choice(list(UNARY_FUNCS.keys()))
        child = _make_leaf()
        return (fn, child, None)
    else:
        fn = random.choice(list(BIN_FUNCS.keys()))
        left = _make_leaf()
        right = _make_leaf()
        return (fn, left, right)

def _random_tree(max_depth=4, depth=0):
    if depth >= max_depth:
        return _make_leaf()
    return _make_node()

def _eval_tree(tree, X):
    name, left, right = tree
    if left is None and right is None:
        return TERM_FUNCS[name](X, X)
    if right is None:
        # unary
        a = _eval_tree(left, X)
        return UNARY_FUNCS[name](a, X)
    a = _eval_tree(left, X)
    b = _eval_tree(right, X)
    return BIN_FUNCS[name](a, X)

def _tree_depth(tree):
    name, left, right = tree
    if left is None and right is None:
        return 0
    if right is None:
        return 1 + _tree_depth(left)
    return 1 + max(_tree_depth(left), _tree_depth(right))

def _subtrees(tree):
    """Yield (parent_index, which_child) for mutation points."""
    results = []
    def rec(node, idx, parent, which):
        results.append((parent, which, idx))
        name, left, right = node
        if left is not None:
            rec(left, idx*2+1, idx, 'left')
        if right is not None:
            rec(right, idx*2+2, idx, 'right')
    rec(tree, 0, -1, None)
    return results

def _clone_tree(tree):
    name, left, right = tree
    if left is None and right is None:
        return (name, None, None)
    if right is None:
        return (name, _clone_tree(left), None)
    return (name, _clone_tree(left), _clone_tree(right))

def _mutate(tree, max_depth=5):
    new_tree = _clone_tree(tree)
    nodes = _subtrees(new_tree)
    if not nodes:
        return new_tree
    # pick a random node to replace
    parent_idx, which, idx = random.choice(nodes)
    # replace the subtree at that node
    new_sub = _random_tree(max_depth=max(1, max_depth - _tree_depth(new_tree)), depth=0)
    # we need to set the child pointer
    # rebuild: simplest is to do a targeted replace
    def replace(node, target_idx, new_node):
        name, left, right = node
        if left is None and right is None:
            return new_node
        if right is None:
            return (name, replace(left, target_idx, new_node), None)
        return (name, replace(left, target_idx, new_node), replace(right, target_idx, new_node))
    # The idx in _subtrees is a heap index; implement replace by heap index
    def replace_by_heap(node, heap_idx, new_node):
        if node is None:
            return node
        if heap_idx == 0:
            return new_node
        name, left, right = node
        if left is not None:
            left = replace_by_heap(left, heap_idx*2+1, new_node)
        if right is not None:
            right = replace_by_heap(right, heap_idx*2+2, new_node)
        return (name, left, right)
    new_tree = replace_by_heap(new_tree, idx, new_sub)
    return new_tree

def _crossover(t1, t2, max_depth=5):
    t1 = _clone_tree(t1)
    t2 = _clone_tree(t2)
    nodes1 = _subtrees(t1)
    nodes2 = _subtrees(t2)
    if not nodes1 or not nodes2:
        return t1
    _, _, idx1 = random.choice(nodes1)
    _, _, idx2 = random.choice(nodes2)
    # extract subtree at idx2 from t2
    def extract_by_heap(node, heap_idx):
        if node is None:
            return None
        if heap_idx == 0:
            return node
        name, left, right = node
        if left is not None:
            left = extract_by_heap(left, heap_idx*2+1)
        if right is not None:
            right = extract_by_heap(right, heap_idx*2+2)
        return (name, left, right)
    sub2 = extract_by_heap(t2, idx2)
    # replace subtree at idx1 in t1 with sub2
    def replace_by_heap(node, heap_idx, new_node):
        if node is None:
            return node
        if heap_idx == 0:
            return new_node
        name, left, right = node
        if left is not None:
            left = replace_by_heap(left, heap_idx*2+1, new_node)
        if right is not None:
            right = replace_by_heap(right, heap_idx*2+2, new_node)
        return (name, left, right)
    t1 = replace_by_heap(t1, idx1, sub2)
    return t1

# ---------- Gaussian Process (RBF kernel) baseline ----------
def _fit_gp(X_train, y_train, time_budget=20.0):
    """Fit a GP with RBF kernel; return predict function."""
    start = time.time()
    n = X_train.shape[0]
    x = X_train.ravel()
    y = y_train.ravel()

    # Precompute squared distances
    sq = x ** 2
    D = sq[:, None] + sq[None, :] - 2 * np.outer(x, x)
    np.maximum(D, 0, out=D)

    # Length scales to try
    length_scales = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    noises = [1e-8, 1e-6, 1e-4]

    best_score = np.inf
    best_params = None

    for ls in length_scales:
        if time.time() - start > time_budget:
            break
        sigma2 = ls ** 2
        K = np.exp(-D / (2 * sigma2))
        for noise in noises:
            if time.time() - start > time_budget:
                break
            try:
                A = K + noise * np.eye(n)
                L = np.linalg.cholesky(A)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                # LOO prediction (efficient for GP)
                v = np.linalg.solve(L, np.ones(n))
                loo_pred = y - v * alpha / (np.diag(np.linalg.inv(A)) if False else (1.0 / np.einsum('ij,ij->i', np.linalg.cho_solve((L, True), np.eye(n)))))
                # Simpler: LOO via standard formula
                # k_i = K[i,i] = 1
                # loo_i = (y_i - v_i * alpha_i) / (1 - v_i * (L^-1 ... )) -- use standard
                # Use: loo_i = y_i - (v_i * alpha_i) / (k_ii - v_i^T L^-1 v_i)
                # k_ii = 1
                # Compute w = L^-1 v
                w = np.linalg.solve(L, v)
                denom = 1.0 - np.sum(w * v, axis=1)
                np.maximum(denom, 1e-10, out=denom)
                loo_pred = y - (v * alpha) / denom
                loo_mse = np.mean((loo_pred - y) ** 2)
                if loo_mse < best_score:
                    best_score = loo_mse
                    best_params = (ls, noise, L, alpha)
            except np.linalg.LinAlgError:
                continue

    if best_params is None:
        # fallback: simple mean
        m = float(np.mean(y))
        return lambda X: np.full(X.shape[0], m)

    ls, noise, L, alpha = best_params
    sigma2 = ls ** 2

    def predict(X):
        xt = X.ravel()
        Dtest = xt[:, None] ** 2 + sq[None, :] - 2 * np.outer(xt, x)
        np.maximum(Dtest, 0, out=Dtest)
        Ktest = np.exp(-Dtest / (2 * sigma2))
        return Ktest @ alpha

    return predict

# ---------- Main fit ----------
def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 330.0
    MARGIN = 20.0  # seconds to spare

    def _time_left():
        return DEADLINE - (time.time() - start_time) - MARGIN

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n = X_train.shape[0]
    x = X_train.ravel()
    y = y_train

    # --- Baseline: GP ---
    gp_predict = None
    gp_loo = np.inf
    try:
        gp_predict = _fit_gp(X_train, y_train, time_budget=15.0)
        # LOO for GP
        gp_pred_train = gp_predict(X_train)
        gp_loo = np.mean((gp_pred_train - y) ** 2)
    except Exception:
        gp_predict = None
        gp_loo = np.inf

    # --- Symbolic regression via evolutionary search ---
    best_expr = None
    best_expr_loo = np.inf
    best_expr_pred = None

    rng = random.Random(42)
    np.random.seed(42)

    # Population init
    pop_size = 40
    population = []
    for i in range(pop_size):
        depth = rng.randint(2, 5)
        tree = _random_tree(max_depth=depth)
        population.append(tree)

    # Also seed with some hand-crafted expressions
    handcrafted = [
        ('add', ('x', None, None), ('1', None, None)),
        ('mul', ('x', None, None), ('x', None, None)),
        ('sin', ('x', None, None), None),
        ('cos', ('x', None, None), None),
        ('exp', ('x', None, None), None),
        ('sqrt', ('x', None, None), None),
        ('div', ('x', None, None), ('add', ('x', None, None), ('1', None, None))),
        ('mul', ('sin', ('x', None, None), None), ('cos', ('x', None, None), None)),
    ]
    for hc in handcrafted:
        population.append(_clone_tree(hc))

    def _loo_mse_for_tree(tree):
        try:
            pred = _eval_tree(tree, x)
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y) ** 2)
            return mse
        except Exception:
            return np.inf

    # Evaluate initial population
    for i, tree in enumerate(population):
        mse = _loo_mse_for_tree(tree)
        if mse < best_expr_loo:
            best_expr_loo = mse
            best_expr = _clone_tree(tree)

    # Evolutionary loop
    gen = 0
    max_gens = 200
    while gen < max_gens and _time_left() > 0.5:
        gen += 1
        # Evaluate all
        scored = []
        for tree in population:
            mse = _loo_mse_for_tree(tree)
            scored.append((mse, tree))
        scored.sort(key=lambda t: t[0])
        if scored[0][0] < best_expr_loo:
            best_expr_loo = scored[0][0]
            best_expr = _clone_tree(scored[0][1])

        # Selection: keep top half
        top = [t for _, t in scored[:pop_size // 2]]
        new_pop = [_clone_tree(t) for t in top]

        # Fill rest with mutations and crossovers
        while len(new_pop) < pop_size and _time_left() > 0.3:
            r = random.random()
            if r < 0.5 and len(top) >= 2:
                # crossover
                t1 = random.choice(top)
                t2 = random.choice(top)
                child = _crossover(t1, t2, max_depth=5)
            elif r < 0.85:
                # mutation
                t1 = random.choice(top)
                child = _mutate(t1, max_depth=5)
            else:
                # random
                child = _random_tree(max_depth=rng.randint(2, 5))
            new_pop.append(child)

        population = new_pop

    # Evaluate best expression
    if best_expr is not None:
        try:
            best_expr_pred = _eval_tree(best_expr, x)
            if not np.all(np.isfinite(best_expr_pred)):
                best_expr_pred = None
        except Exception:
            best_expr_pred = None

    # --- Select final model ---
    # Compare GP vs best expression on training (proxy for LOO)
    if best_expr_pred is not None and np.all(np.isfinite(best_expr_pred)):
        expr_train_mse = np.mean((best_expr_pred - y) ** 2)
    else:
        expr_train_mse = np.inf

    use_expr = (expr_train_mse < gp_loo) and (best_expr_pred is not None)

    if use_expr:
        def predict(X):
            Xt = np.asarray(X, dtype=float)
            try:
                p = _eval_tree(best_expr, Xt.ravel())
                p = np.where(np.isfinite(p), p, 0.0)
                return p.reshape(Xt.shape[0])
            except Exception:
                return np.zeros(Xt.shape[0])
    else:
        if gp_predict is not None:
            def predict(X):
                Xt = np.asarray(X, dtype=float)
                try:
                    p = gp_predict(Xt)
                    p = np.where(np.isfinite(p), p, 0.0)
                    return p.reshape(Xt.shape[0])
                except Exception:
                    return np.zeros(Xt.shape[0])
        else:
            m = float(np.mean(y))
            def predict(X):
                Xt = np.asarray(X, dtype=float)
                return np.full(Xt.shape[0], m)

    return predict

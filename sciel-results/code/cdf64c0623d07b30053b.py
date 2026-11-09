"""Symbolic regression over 4 hidden 1-D noiseless equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0 - 15.0  # safety margin
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _cv_score(fit_fn, X, y, k=5):
    """Return mean CV normalized-MSE (lower better) for a fit function.
    fit_fn(Xtr, ytr) -> predict callable."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if len(tr) < 3 or len(f) < 1:
            return 1e18
        try:
            pred = fit_fn(X[tr], y[tr])
            p = np.asarray(pred(X[f]), dtype=float).ravel()
            if not np.all(np.isfinite(p)):
                return 1e18
            var = np.var(y[f])
            if var < 1e-12:
                mse = np.mean((p - y[f]) ** 2)
            else:
                mse = np.mean((p - y[f]) ** 2) / var
            scores.append(mse)
        except Exception:
            return 1e18
    return float(np.mean(scores))


def _poly_fit(deg):
    def fit_fn(X, y):
        Xc = X.ravel()
        w, *_ = np.linalg.lstsq(np.vander(Xc, deg + 1, increasing=True), y, rcond=None)
        return lambda T: np.vander(T.ravel(), deg + 1, increasing=True) @ w
    return fit_fn


def _rbf_fit(gamma, alpha=1e-6):
    def fit_fn(X, y):
        Xc = X.ravel()
        # RBF kernel matrix
        d2 = (Xc[:, None] - Xc[None, :]) ** 2
        K = np.exp(-gamma * d2) + alpha * np.eye(len(Xc))
        w, *_ = np.linalg.lstsq(K, y, rcond=None)
        def predict(T):
            Tc = T.ravel()
            Kt = np.exp(-gamma * (Tc[:, None] - Xc[None, :]) ** 2)
            return Kt @ w
        return predict
    return fit_fn


def _make_gp_predict(tree, Xs, ys):
    """tree: nested list. Build prediction vector given train Xs, ys."""
    def eval_node(node, X):
        if isinstance(node, tuple):
            op = node[0]
            if op == "add":
                return eval_node(node[1], X) + eval_node(node[2], X)
            if op == "sub":
                return eval_node(node[1], X) - eval_node(node[2], X)
            if op == "mul":
                return eval_node(node[1], X) * eval_node(node[2], X)
            if op == "div":
                a = eval_node(node[1], X)
                b = eval_node(node[2], X)
                return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1.0), 0.0)
            if op == "sin":
                return np.sin(eval_node(node[1], X))
            if op == "cos":
                return np.cos(eval_node(node[1], X))
            if op == "exp":
                return np.exp(np.clip(eval_node(node[1], X), -50, 50))
            if op == "sqrt":
                return np.sqrt(np.clip(eval_node(node[1], X), 0, None))
            if op == "abs":
                return np.abs(eval_node(node[1], X))
        else:
            kind, val = node
            if kind == "x":
                return X
            if kind == "c":
                return np.full(len(X), val)
            if kind == "const":
                return np.full(len(X), val)
        return np.zeros(len(X))

    def predict(T):
        Tc = T.ravel().astype(float)
        return eval_node(tree, Tc)
    return predict


def _gp_search(X, y, budget):
    """Lightweight genetic programming over a small function set.
    Returns (tree, cv_score) of best found, or (None, inf)."""
    rng = np.random.RandomState(1)
    Xc = X.ravel().astype(float)
    n = len(Xc)

    # candidate constants
    const_vals = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
                  float(np.mean(y)), float(np.median(y)),
                  float(np.max(y)), float(np.min(y))]
    const_vals = list(set(const_vals))

    def rand_const():
        return ("const", float(rng.choice(const_vals)))

    def rand_leaf():
        if rng.rand() < 0.6:
            return ("x", 0)
        return rand_const()

    def rand_tree(depth):
        if depth <= 0:
            return rand_leaf()
        r = rng.rand()
        if r < 0.5:
            return rand_leaf()
        ops = ["add", "sub", "mul", "div", "sin", "cos", "exp", "sqrt", "abs"]
        op = ops[int(rng.randint(len(ops)))]
        if op in ("sin", "cos", "exp", "sqrt", "abs"):
            return (op, rand_tree(depth - 1))
        return (op, rand_tree(depth - 1), rand_tree(depth - 1))

    def tree_size(t):
        if isinstance(t, tuple):
            return 1 + sum(tree_size(c) for c in t[1:])
        return 1

    def cv_of(tree):
        pred = _make_gp_predict(tree, Xc, y)
        return _cv_score(lambda Xtr, ytr: (lambda T: pred(T)), X, y, k=5)

    # initialize population
    pop = [rand_tree(rng.randint(2, 5)) for _ in range(40)]
    best = None
    best_score = np.inf
    for t in pop:
        s = cv_of(t)
        if s < best_score:
            best_score = s
            best = t

    # evolve
    generations = 0
    while _time_left() > 20 and generations < 60:
        generations += 1
        # evaluate current pop
        scored = []
        for t in pop:
            scored.append((cv_of(t), t))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best = scored[0][1]
        if best_score < 1e-6:
            break
        # next gen: elitism + crossover/mutation
        next_pop = [scored[i][1] for i in range(min(8, len(scored)))]
        while len(next_pop) < len(pop):
            p1 = scored[int(rng.randint(len(scored)))]
            p2 = scored[int(rng.randint(len(scored)))]
            if rng.rand() < 0.5:
                # crossover
                child = _crossover(p1[1], p2[1], rng)
            else:
                child = _mutate(p1[1], rng, rand_tree)
            next_pop.append(child)
        pop = next_pop

    return best, best_score


def _crossover(a, b, rng):
    # simple subtree swap
    if rng.rand() < 0.3:
        return _copy(a)
    # pick a random node in a to replace with a random subtree from b
    return _swap_random(a, _copy(b), rng)


def _copy(t):
    if isinstance(t, tuple):
        return (t[0], *[_copy(c) for c in t[1:]])
    return t


def _mutate(t, rng, rand_tree):
    if rng.rand() < 0.2:
        return rand_tree(rng.randint(1, 3))
    if isinstance(t, tuple):
        children = list(t[1:])
        i = int(rng.randint(len(children)))
        children[i] = _mutate(children[i], rng, rand_tree)
        return (t[0], *children)
    return t


def _swap_random(a, b, rng):
    # replace a random leaf/subtree in a with b (if sizes allow)
    # simple: with some prob replace a whole subtree
    if rng.rand() < 0.5:
        return b
    if isinstance(a, tuple):
        children = list(a[1:])
        i = int(rng.randint(len(children)))
        children[i] = _swap_random(children[i], b, rng)
        return (a[0], *children)
    return b


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xc = X.ravel().astype(float)
    n = len(Xc)

    # baseline: mean predictor
    mean_pred = float(np.mean(y))

    # Strategy pool: (name, fit_fn, cv_score)
    candidates = []

    # Polynomials of various degrees
    for deg in [1, 2, 3, 4, 5, 6, 8]:
        if _time_left() < 15:
            break
        fn = _poly_fit(deg)
        s = _cv_score(fn, X, y, k=5)
        candidates.append((deg, s, fn))

    # RBF with various gammas
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if _time_left() < 15:
            break
        fn = _rbf_fit(gamma)
        s = _cv_score(fn, X, y, k=5)
        candidates.append(("rbf_%g" % gamma, s, fn))

    # Genetic programming
    if _time_left() > 30:
        tree, s = _gp_search(X, y, _time_left())
        if tree is not None:
            pred = _make_gp_predict(tree, Xc, y)
            fn = lambda Xtr, ytr: (lambda T: pred(T))
            candidates.append(("gp", s, fn))

    # Pick best by CV score; fall back to mean if all bad
    if not candidates:
        return lambda T: np.full(len(np.asarray(T).ravel()), mean_pred)

    candidates.sort(key=lambda c: c[1])
    best_score, best_fn = candidates[0][1], candidates[0][2]

    # If best CV is terrible, prefer a low-degree polynomial or mean
    if best_score > 1.0:
        # find best polynomial
        poly_cands = [c for c in candidates if isinstance(c[0], int)]
        if poly_cands:
            poly_cands.sort(key=lambda c: c[1])
            best_fn = poly_cands[0][2]
            best_score = poly_cands[0][1]
        if best_score > 5.0:
            # use mean
            return lambda T: np.full(len(np.asarray(T).ravel()), mean_pred)

    # Final fit on all data
    try:
        final_predict = best_fn(X, y)
        # sanity check on training
        p = np.asarray(final_predict(X), dtype=float).ravel()
        if not np.all(np.isfinite(p)):
            raise ValueError("non-finite")
        return final_predict
    except Exception:
        # fallback: degree-2 polynomial
        try:
            w, *_ = np.linalg.lstsq(np.vander(Xc, 3, increasing=True), y, rcond=None)
            return lambda T: np.vander(np.asarray(T).ravel(), 3, increasing=True) @ w
        except Exception:
            return lambda T: np.full(len(np.asarray(T).ravel()), mean_pred)
# EVOLVE-BLOCK-END

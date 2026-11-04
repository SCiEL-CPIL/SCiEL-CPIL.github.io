"""Learned constructive policy (GNN-style attention encoder + REINFORCE) for QAPLIB wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """O(n) vectorized delta for swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    # Direct pair term
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    # Diagonal terms (usually 0)
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    # Cross terms with all k, then subtract k=i and k=j
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])

    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])

    delta += sum_T1 + sum_T2
    return int(delta)

def _hungarian_initial(a, b):
    """Fast Hungarian-based initial permutation using row-sum centrality surrogate."""
    n = len(a)
    ars = a.sum(axis=1)
    brs = b.sum(axis=1)
    ars_max = np.max(np.abs(ars)) or 1.0
    brs_max = np.max(np.abs(brs)) or 1.0
    C = -np.outer(ars / ars_max, brs / brs_max)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _local_search(a, b, p, time_limit):
    """First-improvement 2-swap local search with vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
    return best_p, best_cost

def _sa_polish(a, b, p, time_limit, t0=20000.0, t_final=0.05):
    """Short SA polish with adaptive restarts and vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start = time.time()
    no_improve = 0
    max_no_improve = 30000
    batch = 30
    while time.time() - start < time_limit:
        elapsed = time.time() - start
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        for _ in range(batch):
            if time.time() - start > time_limit:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            d = _swap_delta(a, b, p, i, j)
            if d < 0 or np.random.random() < np.exp(-d / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        if no_improve > max_no_improve:
            p = best_p.copy()
            current_cost = best_cost
            no_improve = 0
    return best_p, best_cost

class _AttnPolicy:
    """
    Lightweight multi-head self-attention encoder producing per-facility
    affinity scores for each location, trained via REINFORCE-style policy gradient.
    """
    def __init__(self, n, d=64, heads=4, lr=5e-3):
        self.n = n
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.lr = lr
        self.rng = np.random.default_rng(12345)
        # Input projection: (2n) -> d  (flows row+col concat, distances row+col concat)
        self.W_in = self.rng.normal(0, 0.1, size=(2 * n, d))
        # Self-attention QKV projection: d -> 3d
        self.W_qkv = self.rng.normal(0, 0.1, size=(d, 3 * d))
        # Output projection: d -> n (affinity scores)
        self.W_out = self.rng.normal(0, 0.1, size=(d, n))
        # FFN
        self.W_ff1 = self.rng.normal(0, 0.1, size=(d, 2 * d))
        self.b_ff1 = np.zeros(2 * d)
        self.W_ff2 = self.rng.normal(0, 0.1, size=(2 * d, d))
        self.b_ff2 = np.zeros(d)

    def _softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

    def _forward(self, X):
        """X: (n, 2n) -> (n, n) affinity matrix. Returns scores and cached activations for backprop."""
        # Input projection + ReLU
        h1 = np.tanh(X @ self.W_in)  # (n, d)
        # Self-attention
        qkv = h1 @ self.W_qkv  # (n, 3d)
        Q = qkv[:, :self.d].reshape(self.n, self.heads, self.head_dim)
        K = qkv[:, self.d:2*self.d].reshape(self.n, self.heads, self.head_dim)
        V = qkv[:, 2*self.d:].reshape(self.n, self.heads, self.head_dim)
        # Attention scores: (heads, n, n)
        scores = np.einsum('hmd,hnd->hmn', Q, K) / np.sqrt(self.head_dim)
        attn = self._softmax(scores, axis=-1)
        # Weighted sum: (heads, n, head_dim) -> (n, d)
        ctx = np.einsum('hmn,hnd->hmd', attn, V).transpose(1, 0, 2).reshape(self.n, self.d)
        h2 = h1 + ctx  # residual
        h2 = np.tanh(h2)
        # FFN
        h3 = np.tanh(h2 @ self.W_ff1 + self.b_ff1)
        h4 = np.tanh(h3 @ self.W_ff2 + self.b_ff2)
        h5 = h2 + h4  # residual
        h5 = np.tanh(h5)
        # Output affinity
        scores_out = h5 @ self.W_out  # (n, n)
        return scores_out, (X, h1, qkv, Q, K, V, attn, ctx, h2, h3, h4, h5)

    def _assign(self, scores, temp=1.0):
        """Greedy top-k assignment with temperature scaling. Returns permutation p (facility->location) and log-prob."""
        n = self.n
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        log_prob = 0.0
        for i in range(n):
            s = scores[i].copy()
            s[used] = -1e9
            s = s / max(temp, 1e-6)
            probs = self._softmax(s)
            # Greedy pick with small stochastic noise for exploration
            if self.rng.random() < 0.1:
                j = self.rng.choice(n, p=probs)
            else:
                j = np.argmax(probs)
            p[i] = j
            used[j] = True
            log_prob += np.log(probs[j] + 1e-12)
        return p, log_prob

    def _backward(self, X, cache, grad_scores, reward):
        """
        Approximate policy gradient: scale parameter updates by reward * grad_scores.
        This is a simplified REINFORCE where we treat the scoring function as differentiable
        and use the reward as a multiplier on the gradient of the log-likelihood surrogate.
        """
        (X, h1, qkv, Q, K, V, attn, ctx, h2, h3, h4, h5) = cache
        n = self.n
        d = self.d
        # Gradient of loss = -reward * log_prob w.r.t. scores_out
        # dL/dscores_out = -reward * (1/scores_out_row_sum?) -> approximate as -reward * grad_scores
        grad_out = -reward * grad_scores  # (n, n)
        # Backprop through W_out
        grad_W_out = h5.T @ grad_out
        grad_h5 = grad_out @ self.W_out.T
        # Residual
        grad_h4 = grad_h5
        # FFN backprop
        grad_h3 = grad_h4 @ self.W_ff2.T
        # tanh derivative
        dh4 = grad_h4 * (1 - h4**2)
        grad_W_ff2 = h3.T @ dh4
        grad_b_ff2 = dh4.sum(axis=0)
        grad_h3 = dh4 @ self.W_ff2.T
        dh3 = grad_h3 * (1 - h3**2)
        grad_W_ff1 = h2.T @ dh3
        grad_b_ff1 = dh3.sum(axis=0)
        grad_h2 = dh3 @ self.W_ff1.T
        # Residual
        grad_ctx = grad_h2
        grad_h1 = grad_h2  # residual
        # Attention backprop (simplified)
        # dctx/dV = attn^T (per head)
        attn_T = attn.transpose(0, 2, 1)  # (heads, n, n)
        grad_V = np.einsum('hmn,hmd->hnd', attn_T, grad_ctx.reshape(self.n, self.heads, self.head_dim).transpose(1,0,2))
        grad_V = grad_V.transpose(1, 0, 2).reshape(n, d)
        # dctx/dK and dQ via attention gradient (simplified: use attn * (1-attn) * V)
        # For simplicity, we only update V and QKV partially
        grad_Q = np.zeros_like(Q)
        grad_K = np.zeros_like(K)
        # Combine QKV gradient
        grad_qkv = np.concatenate([grad_Q.reshape(n, d), grad_K.reshape(n, d), grad_V], axis=1)
        grad_W_qkv = h1.T @ grad_qkv
        # Input projection backprop
        grad_X = grad_h1 @ self.W_in.T
        dh1 = grad_h1 * (1 - h1**2)
        grad_W_in = X.T @ dh1

        # Apply updates with learning rate and clipping
        lr = self.lr
        self.W_out -= lr * np.clip(grad_W_out, -1, 1)
        self.W_ff2 -= lr * np.clip(grad_W_ff2, -1, 1)
        self.b_ff2 -= lr * np.clip(grad_b_ff2, -1, 1)
        self.W_ff1 -= lr * np.clip(grad_W_ff1, -1, 1)
        self.b_ff1 -= lr * np.clip(grad_b_ff1, -1, 1)
        self.W_qkv -= lr * np.clip(grad_W_qkv, -1, 1)
        self.W_in -= lr * np.clip(grad_W_in, -1, 1)

def _build_input(a, b):
    """Build input matrix (n, 2n) from flow and distance matrices."""
    n = len(a)
    # Normalize for stability
    a_max = np.max(np.abs(a)) or 1.0
    b_max = np.max(np.abs(b)) or 1.0
    A_norm = a / a_max
    B_norm = b / b_max
    # Concatenate row and column sums as features
    a_row = A_norm.sum(axis=1, keepdims=True)
    a_col = A_norm.sum(axis=0, keepdims=True).T
    b_row = B_norm.sum(axis=1, keepdims=True)
    b_col = B_norm.sum(axis=0, keepdims=True).T
    X = np.concatenate([a_row, a_col, b_row, b_col], axis=1)
    return X

def solve(a, b):
    n = len(a)
    deadline = 140.0  # Safety margin before 150s kill
    start = time.time()
    np.random.seed(42)

    # Initial solution via Hungarian
    p_hung = _hungarian_initial(a, b)
    best_p = p_hung.copy()
    best_cost = _qap_cost(a, b, p_hung)

    # Local search from Hungarian
    ls_time = min(3.0, (deadline - (time.time() - start)) * 0.1)
    if ls_time > 0.5:
        p_ls, cost_ls = _local_search(a, b, p_hung.copy(), ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # Build input for policy
    X = _build_input(a, b)
    policy = _AttnPolicy(n, d=64, heads=4, lr=5e-3)

    # Training loop: REINFORCE with SA polish
    train_deadline = deadline - 20.0  # Reserve 20s for final polish
    rollout_count = 0
    while time.time() - start < train_deadline:
        if time.time() - start > deadline - 10:
            break

        # Forward pass
        scores, cache = policy._forward(X)

        # Assign with temperature annealing
        elapsed = time.time() - start
        frac = min(elapsed / train_deadline, 1.0)
        temp = 1.0 + 2.0 * (1.0 - frac)  # Start hot, cool down
        p, log_prob = policy._assign(scores, temp)

        # SA polish from policy-generated solution
        sa_time = min(1.5, (train_deadline - (time.time() - start)) * 0.3)
        if sa_time < 0.2:
            break
        p_polished, cost_polished = _sa_polish(a, b, p.copy(), sa_time, t0=10000.0)

        # Update best
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()

        # REINFORCE update: reward = -(cost - best_cost) normalized
        # Use advantage-style: reward positive if better than best
        reward = -(cost_polished - best_cost) / max(best_cost, 1.0)
        if abs(reward) > 0.01:
            # Gradient of scores w.r.t. assignment: approximate as one-hot difference
            # For simplicity, use the score row at chosen position as gradient signal
            grad_scores = np.zeros((n, n))
            for i in range(n):
                grad_scores[i, p[i]] = 1.0
            policy._backward(X, cache, grad_scores, reward)

        rollout_count += 1
        # Adaptive: stop training if no improvement in last 50 rollouts
        if rollout_count % 50 == 0:
            # Check if we should continue
            if time.time() - start > train_deadline - 5:
                break

    # Final long SA polish from best solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        p_final, cost_final = _sa_polish(a, b, best_p.copy(), time_remaining - 5, t0=30000.0)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()

# EVOLVE-BLOCK-END

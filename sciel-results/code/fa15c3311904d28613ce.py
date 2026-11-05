import concurrent.futures

def run_milp(c, A, b, N):
    try:
        res = milp(c=c, constraints=LinearConstraint(A, ub=b), integrality=np.ones(N), bounds=Bounds(0, 1))
        if res.success:
            return np.where(res.x > 0.5)[0] + 1
    except:
        pass
    return None

# In construct:
with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(run_milp, c, A, b, N)
    try:
        sol = future.result(timeout=time_limit - (time.time() - start_time))
        if sol is not None and len(sol) > len(best_greedy):
            return list(sol)
    except concurrent.futures.TimeoutError:
        pass

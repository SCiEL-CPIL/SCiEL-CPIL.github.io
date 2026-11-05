import importlib, inspect
m = importlib.import_module("autoresearch_suite.science_tcs_expansion")
src = inspect.getsource(m)
print("LEN", len(src))
print(src)

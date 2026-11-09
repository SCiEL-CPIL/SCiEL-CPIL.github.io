import autoresearch_suite.science_tcs_expansion as m
import inspect
print([x for x in dir(m) if not x.startswith('__')])
print("----")
for name in dir(m):
    if not name.startswith('__'):
        obj = getattr(m, name)
        print(name, type(obj))

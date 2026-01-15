import time
import multiprocessing as mp

def foo(x, y):
    print(f"[{time.time():.2f}] Start task: {x}+{y}")
    time.sleep(1)
    result = x + y
    print(f"[{time.time():.2f}] End task: {x}+{y} = {result}")
    return result

# Test apply
print("=== Using apply (synchronous) ===")
start = time.time()
with mp.Pool(2) as pool:
    a = pool.apply(foo, (1, 2))
    b = pool.apply(foo, (3, 4))
print(f"Total time: {time.time() - start:.2f}s\n")

# Test apply_async
print("=== Using apply_async (asynchronous) ===")
start = time.time()
with mp.Pool(2) as pool:
    h1 = pool.apply_async(foo, (1, 2))
    h2 = pool.apply_async(foo, (3, 4))
    a = h1.get()
    b = h2.get()
print(f"Total time: {time.time() - start:.2f}s")
# Benchmarks

## Simple

### 99 x 99

```
With numpy @ CPU: 4.514676
With cupy @ GPU: 22.247727
```

### 999 x 99

```
With numpy @ CPU: 39.853067
With cupy @ GPU: 22.683342
```

### 10,000 x 10,000

_"After running this script on an Intel Xeon 1240v3 machine with Nvidia Geforce GT1030 GPU accelerator from Cherry Servers GPU Cloud, we’ve confirmed that integer addition runs many times faster on a GPU. For instance, GPU runs integer addition **~1294 times faster** when 10000x10000 matrix is being used."_

## Resources

- [A Complete Introduction to GPU Programming With Practical Examples in CUDA and Python](https://www.cherryservers.com/blog/introduction-to-gpu-programming-with-cuda-and-python)

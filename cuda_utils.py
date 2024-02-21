from torch.utils.cpp_extension import load_inline

cuda_src_prefix = """
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
"""

def load_cuda_inline(cuda_src, cpp_src, funcs, optimize=True, verbose=False, name=None):
    cuda_src = cuda_src_prefix + cuda_src
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if optimize else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs, extra_cuda_cflags=[flags], verbose=verbose, name=name)

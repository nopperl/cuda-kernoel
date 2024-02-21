from torch.utils.cpp_extension import load_inline


def load_cuda_inline(cuda_src, cpp_src, funcs, optimize=True, verbose=False, name=None):
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if optimize else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs, extra_cuda_cflags=[flags], verbose=verbose, name=name)
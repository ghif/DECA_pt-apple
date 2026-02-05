from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

USE_NINJA = os.getenv('USE_NINJA') == '1'

# Always build CPU version
ext_modules = [
    CppExtension('standard_rasterize_cpu', [
        'standard_rasterize_cpu.cpp',
    ])
]

setup(
    name='standard_rasterize',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)}
)

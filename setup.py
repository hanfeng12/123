import subprocess
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

subprocess.check_output(["bash", "generate.sh"])

comp_path = Path(__file__).parent / "comp"
object_files = list(map(str, comp_path.glob("*.a")))

setup(
    name="main",
    verbose=True,
    ext_modules=[
        CppExtension("main", ["main.cpp"], extra_objects=object_files, extra_compile_args=["-std=c++17", "-g"])
    ],
    include_dirs=["Halide/include", str(comp_path)],
    cmdclass={"build_ext": BuildExtension},
)

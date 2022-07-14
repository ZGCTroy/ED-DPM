from setuptools import setup

setup(
    name="entropy_driven_guided-diffusion",
    py_modules=["entropy_driven_guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)

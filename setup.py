from setuptools import setup, find_packages


setup(
    name="project",
    version="0.0.1",
    description="",
    url="http://github.com/intel-comp-saude-ufes/2024-1-P2-classificador-cancer-de-pulmao",
    author="luizcarloscf",
    license="MIT",
    packages=find_packages("."),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "train=project.train:main",
            "test=project.inference:main",
            "split=project.split:main",
        ],
    },
    zip_safe=False,
    install_requires=[
        "numpy==1.26.4",
        "pillow==10.4.0",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "typing_extensions==4.12.2",
        "pandas==2.2.2",
        "scikit-learn==1.5.0",
        "opencv-contrib-python-headless==4.9.0.80",
        "tensorboard==2.17.0",
        "matplotlib==3.9.0",
        "seaborn==0.13.2",
    ],
)

from setuptools import setup, find_packages
import os
from glob import glob

package_name = "path_tracking"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch",
         glob("launch/*.launch.py")),
        (f"share/{package_name}/config",
         glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    zip_safe=True,
    maintainer="Student",
    maintainer_email="student@university.edu",
    description="Path smoothing and trajectory control for differential drive robots.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "trajectory_tracker_node = nodes.trajectory_tracker_node:main",
        ],
    },
)

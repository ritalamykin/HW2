from setuptools import setup, find_packages, Extension

setup(name="kmeans",
      version="1.0.0",
      description="desc",
      author="<your name>",
      author_email="a@b.com",
      ext_modules=[Extension("mykmeanssp",["kmeans.c"])])

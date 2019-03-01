from setuptools import setup
with open("README.md") as fh:
    long_description = fh.read()
    
if __name__ == '__main__':
    setup(name = 'ace_nn',
          version = '0.1',
          description = 'Alternating Conditional Exceptation with Neural Network',
          author = 'zhaofeng-shu33',
          author_email = '616545598@qq.com',
          url = 'https://github.com/zhaofeng-shu33/background_mask',
          maintainer = 'zhaofeng-shu33',
          maintainer_email = '616545598@qq.com',
          long_description = long_description,
          long_description_content_type="text/markdown",          
          install_requires = ['keras', 'tensorflow', 'sklearn'],
          license = 'Apache License Version 2.0',
          py_modules = ['ace_nn'],
          classifiers = (
              "Development Status :: 4 - Beta",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 2.7",
              "Operating System :: OS Independent",
          ),
          )
from setuptools import setup
from setuptools import find_packages


setup(name='DLTK',
      version='1.0.0',
      description='Deep Learning Tool Kit',
      author='Zhijia Hu',
      author_email='z.jia.hu@gmail.com',
      url='https://github.com/zhijiahu/dltk',
      download_url='https://github.com/zhijiahu/dltk/tarball/1.0.0',
      license='MIT',
      install_requires=['click>=6.7',
      ],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
packages=find_packages())

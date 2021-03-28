from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='shadowCut',
  version='0.0.1',
  description='Deep Learning Model Training , Data Augmmentation and Image Resizing',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Aayush Jain',
  author_email='jainaayush99.aj@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  packages=find_packages(),
  install_requires=['PIL', 'os', 'sys', 'tensorflow', 'glob2', 'keras'] 
)
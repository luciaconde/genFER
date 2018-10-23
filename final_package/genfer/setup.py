from setuptools import setup

setup(name='genfer',
      version='0.1',
      description='Generalized CNN-based facial expression recognition',
      url='https://github.com/luciaconde/genFER',
      author='L.Conde',
      packages=['genfer'],
      install_requires=['numpy','os','glob','cv2','sys','argparse','csv','scikit-learn']
)

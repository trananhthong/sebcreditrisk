from setuptools import setup

setup(
	name='sebcreditrisk',
	version='0.0.1',
	keywords='credit risk',
	license='MIT',
	python_requires='>=3.7',
	install_requires=[
		'numpy',
		'scipy',
		'pandas',
		'openpyxl',
        'torch'
	],
	)
# Deep Umbra: A Global-Scale Generative Adversarial Approach for Sunlight Access and Shadow Accumulation in Urban Spaces

URL: http://evl.uic.edu/shadows/

Deep Umbra is a a novel computational framework that enables the quantification of sunlight access and shadows at a global scale. Our framework is based on a generative adversarial network that considers the physical form of cities to compute high-resolution spatial information of accumulated sunlight access for the different seasons of the year. Deep Umbra's primary motivation is the impact that shadow management can have in people's quality of live, since it can affect levels of comfort, heat distribution, public parks, etc.

We also present the Global Shadow Dataset, a comprehensive dataset with the accumulated shadow information for over 100 cities in 6 continents. In order to visualize the data, click [here](http://evl.uic.edu/shadows/map/). To download the data, click [here](http://evl.uic.edu/shadows/).

This repository contains the code for the paper "Deep Umbra: A Global-Scale Generative Adversarial Approach for Sunlight Access and Shadow Accumulation in Urban Spaces".

![Overview of Deep Umbra](overview.png)

Authors:

Kazi Shahrukh Omar (UIC)

Gustavo Moreira (UFF)

Daniel Hodczak (UIC)

Maryam Hosseini (Rutgers / NYU)

Marcos Lage (UFF)

[Fabio Miranda](https://fmiranda.me) (UIC)

**Paper: Arxiv link soon**

## Prerequisites

The code is written in Python. The following Python packages are required:

```
Python 3.x
Tensorflow 2.9
Pandas 1.4.2
Numpy 1.22.4
Geopandas 0.4.0
OpenCV 4.6
pygeos
pyproj
scikit_image
scikit_learn
rasterio
osmium
```

## Structure

The code is stucture as different Jupyter Notebooks. ``01-download-osm-data.ipynb`` downloads OpenStreetMap data. A height map is generated with ``02-generate-elevation-map``, followed by data preparation in ``03-prepare-data.ipynb``, GAN training (``04-GAN-shadow-height-spatial``), evaluation (``05-evaluate-spatial.ipynb``, ``06-evaluation-all-cases.ipynb``, ``07-evaluation-measurements.ipynb``) and computation of data and performance metrics for multiple cities (``08-compute-cities.ipynb``, ``09-compute-urban-metrics.ipynb ``, ``10-urban-metrics-analysis.ipynb``).

The weights for our pre-trained model can be downloaded [here](https://drive.google.com/file/d/1OumDM4AtiCLjHdHFZOs8rFcEoR3h2rT3/view?usp=sharing). If you use the weights, you can skip the GAN training (``04-GAN-shadow-height-spatial``) and evaluation (``05-evaluate-spatial.ipynb``, ``06-evaluation-all-cases.ipynb``, ``07-evaluation-measurements.ipynb``) steps, and focus on data preparation and inference (remaining steps).

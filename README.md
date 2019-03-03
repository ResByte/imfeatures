imfeatures
===============================

version number: 0.0.1
author: Abhinav Dadhich

Overview
--------

Minimalistic python package to extract deep learning features from a wide variety of pretrained models in Pytorch.


Installation / Usage
--------------------

Dependencies:
- pytorch (v >= 1.0.0)
- torchvision


To install use pip:

    $ pip install imfeatures


Or clone the repo:

    $ git clone https://github.com/resbyte/imfeatures.git
    $ python setup.py install
    

Example
-------

- Imports 

```python
import imfeatures
import torch
```

- create feature extractor, here `resnet50`, with pretrained weights 

```python
feature_extractor = imfeatures.Features('resnet50',pretrained=True)
```

- random image of size `224x224x3` 

```python
x = torch.randn([1,3,224,224])
```

- features

```python 
features = feature_extractor(x)
print(features.shape)
```

Output features will be of shape : `[1, 2048, 1, 1]`
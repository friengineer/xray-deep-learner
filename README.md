# DenseNet on MURA and LERA Datasets using PyTorch

A PyTorch implementation of a 169 layer [DenseNet](https://arxiv.org/abs/1608.06993) model trained on the MURA and LERA datasets, inspired from the papers [arXiv:1712.06957v4](https://arxiv.org/abs/1712.06957) by Pranav Rajpurkar et al. and ['Automated abnormality detection in lower extremity radiographs using deep learning'](https://doi.org/10.1038/s42256-019-0126-0) by Varma et al. MURA is a large dataset of upper extremity musculoskeletal radiographs, where each study is manually labelled by radiologists as either normal or abnormal. LERA is a similar dataset except it contains lower extremity musculoskeletal radiographs. Visit the respective links to learn more about [MURA](https://stanfordmlgroup.github.io/competitions/mura/) and [LERA](https://aimi.stanford.edu/lera-lower-extremity-radiographs-2).

## Important Points:
* This model has been implemented for research purposes only and should not be used in a commercial or clinical environment to provide diagnoses to patients
* The implemented model is a 169 layer DenseNet with a single node output layer initialised with weights from a model pretrained on the ImageNet dataset by default. The user can specify if they would like to use a previously trained model instead.
* Before feeding the images to the network, each image is normalized to have the same mean and standard deviation as the images in the ImageNet training set, scaled to 224 x 224 and augmented with random lateral inversions and rotations.
* The model uses a modified binary cross entropy loss function as mentioned in the MURA paper.
* The learning rate decays by a factor of 10 every time the validation loss plateaus after an epoch.
* The optimisation algorithm used is Adam with default parameters β1 = 0.9 and β2 = 0.999.

According to the MURA paper:

> The model takes as input one or more views for a study of an upper extremity. On each view, our 169-layer convolutional neural network predicts the probability of abnormality. We compute the overall probability of abnormality for the study by taking the arithmetic mean of the abnormality probabilities output by the network for each image.

The model takes as input 'all' the views for a study of an upper or lower extremity. On each view the model predicts the probability of abnormality. The model computes the overall probability of abnormality for the study by taking the arithmetic mean of the abnormality probabilities output by the network for each image. This can be extended to use the entirety of MURA or LERA.

## Instructions

Install dependencies:
* Python3
* PyTorch
* TorchVision
* Numpy
* Pandas
* tqdm
* Matplotlib
* Scikit-Learn
* torchnet

The [requirements file](requirements.txt) can be used to install all dependencies except Python3 as this will need to be installed manually first.

Train the model with `python main.py {study_type}` where {study_type} is either one of the study types in MURA or LERA (e.g. `python main.py wrist`), 'mura' to train the model using all the MURA study types or 'lera' to train the model using all the LERA study types. Run `python main.py -h` to see the study types that can be specified.

Transfer learning can be used to warmstart the model by loading a previously saved model's parameters using the argument `-t {file}` where {file} is the filename of the model saved in the models directory (e.g. `python main.py wrist -t model.pt`).


This project is based off of code created by [Rishabh Agrahari](https://github.com/pyaf/DenseNet-MURA-PyTorch).

## MURA citation
    @ARTICLE{2017arXiv171206957R,
       author = {{Rajpurkar}, P. and {Irvin}, J. and {Bagul}, A. and {Ding}, D. and
      {Duan}, T. and {Mehta}, H. and {Yang}, B. and {Zhu}, K. and
      {Laird}, D. and {Ball}, R.~L. and {Langlotz}, C. and {Shpanskaya}, K. and
      {Lungren}, M.~P. and {Ng}, A.},
        title = "{MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1712.06957},
     primaryClass = "physics.med-ph",
     keywords = {Physics - Medical Physics, Computer Science - Artificial Intelligence},
         year = 2017,
        month = dec,
       adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171206957R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

## LERA citation
    @ARTICLE{VarmaMaya2019Aadi,
    author = {Varma, Maya and Lu, Mandy and Gardner, Rachel and Dunnmon, Jared and Khandwala, Nishith and Rajpurkar, Pranav and Long, Jin and Beaulieu, Christopher and Shpanskaya, Katie and Fei-Fei, Li and Lungren, Matthew P and Patel, Bhavik N},
    journal = {Nature machine intelligence},
    language = {eng},
    number = {12},
    pages = {578-583},
    title = {Automated abnormality detection in lower extremity radiographs using deep learning},
    volume = {1},
    year = {2019},
    issn = {2522-5839},
    }

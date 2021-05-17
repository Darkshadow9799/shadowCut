# shadowCut

* This is a python based library for fastening the deep learning coding process. Programmer has to only select the path for train data, val data and model as mandatory but rest of the parameters are optional and for customized as needed.

* All the tensorflow functionalities are Inherited in this library and major part of Library is constructed using tensorflow library.

* Installation command: ```pip install shadowCut```

* Now lets see at example for using pretrained model:

* Here suppose we have train_path as ```'./train'```, valid_path as ```'./test'```, and the model as ```ResNet152V2(input_shape=[224,224]+[3],weights='imagenet',include_top=False)```
* We can use this package by using:   
```shadowCut.preTrainedModel(train_path=train_path, valid_path=test_path, model=model)```   
* This will return model's history and model respectively.


* **Libraries used:** ```PIL, os, sys, tensorflow, glob2```



## To Get Started:


1. Import the library: ```import shadowCut as sc```
2. Get the training folder address and testing folder address. For example: ```train = './data/train'``` and ```test = './data/test'```
3. If want to get pretrained model: For example want to train ResNet152V2 model
    * ```model = tf.keras.applications.ResNet152V2()```
    * If want custom output model then can be passed else will take already present output model.
    * Config custom options if required.
    * Pass it into: ```sc.preTrainedModel(train = train,
                                     test = test,
                                     model = model,
                                     * other custom parameters)```
4. If want to get custom model: For example want to train ResNet152V2 model
    * ```model = #Custom model```
    * If want custom output model then can be passed else will take already present output model.
    * Config custom options if required.
    * Pass it into:```sc.customTrainedModel(train = train,
                                     test = test,
                                     model = model,
                                     * other custom parameters)```

## For Contribution:

1. General script for all type of Generative Adverserial Network which can be excessed for by any new starter (and also experienced)
2. List of GANs to be added with a general script:
    * DCGAN
    * Condiational GAN
    * Stack GAN
    * Info GAN
    * Disco GAN
    * Style GAN
    * Cycle GAN (more can be added).
3. Add Minmax and Wasserstein loss function.

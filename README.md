# shadowCut

* This is a python based library for fastening the deep learning coding process. Programmer has to only select the path for train data, val data and model as mandatory but rest of the parameters are optional and for customized as needed.

* Now lets see at example for using pretrained model:
* Here suppose we have train_path = './train', valid_path='./test', model=ResNet152V2(input_shape=[224,224]+[3],weights='imagenet',include_top=False).
* Command to pass this in model is shadowCut.preTrainedModel(train_path=train_path, valid_path=test_path, model=model) it will return model's history and model respectively.
* All the tensorflow functionalities are Inherited in this library and major part of Library is constructed using tensorflow library.

* Libraries used: PIL, os, sys, tensorflow, glob2

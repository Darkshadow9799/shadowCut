from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint
from glob import glob

def outputLayer(model, folders):
      x = Flatten()(model.output)
      prediction = Dense(len(folders),activation='softmax')(x)
      return prediction

def preTrainedModel(train_path, valid_path, model, 
                    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], 
                    filepath = 'saved_models/weights-improvement-{epoch:02d}.h5', 
                    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',
                    logfile='my_logs.csv', separator=',', append=False,
                    batch_size = 32, class_mode='categorical', epochs=5, target_size=(256,144),
                    featurewise_center=False, samplewise_center=False,
                    featurewise_std_normalization=False, samplewise_std_normalization=False,
                    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
                    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
                    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                    horizontal_flip=False, vertical_flip=False, rescale=None,
                    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None,
                    test_featurewise_center=False, test_samplewise_center=False,
                    test_featurewise_std_normalization=False, test_samplewise_std_normalization=False,
                    test_zca_whitening=False, test_zca_epsilon=1e-06, test_rotation_range=0, test_width_shift_range=0.0,
                    test_height_shift_range=0.0, test_brightness_range=None, test_shear_range=0.0, test_zoom_range=0.0,
                    test_channel_shift_range=0.0, test_fill_mode='nearest', test_cval=0.0,
                    test_horizontal_flip=False, test_vertical_flip=False, test_rescale=None,
                    test_preprocessing_function=None, test_data_format=None, test_validation_split=0.0, test_dtype=None,
                    validation_freq=1, class_weight=None, max_queue_size=10, workers=1, 
                    use_multiprocessing=False, shuffle=True, initial_epoch=0):
                    
      model = model
                    
      for layers in model.layers:
            layers.trainable=False
                  
      folders=glob(train_path + '/*')

      final_model=Model(inputs=model.input, outputs=outputLayer(model, folders))
                    
      final_model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
                    
      checkpoint = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, mode=mode)
                    
      log_csv = CSVLogger(logfile, separator=separator, append=append)
                    
      callable_list=[checkpoint, log_csv]

      train_datagen = ImageDataGenerator(featurewise_center=featurewise_center, samplewise_center=samplewise_center,
                    featurewise_std_normalization=featurewise_std_normalization, samplewise_std_normalization=samplewise_std_normalization,
                    zca_whitening=zca_whitening, zca_epsilon=zca_epsilon, rotation_range=rotation_range, width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range, brightness_range=brightness_range, shear_range=shear_range, zoom_range=zoom_range,
                    channel_shift_range=channel_shift_range, fill_mode=fill_mode, cval=cval,
                    horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, rescale=rescale,
                    preprocessing_function=preprocessing_function, data_format=data_format, validation_split=validation_split, dtype=dtype)

      test_datagen = ImageDataGenerator(featurewise_center=test_featurewise_center, samplewise_center=test_samplewise_center,
                    featurewise_std_normalization=test_featurewise_std_normalization, samplewise_std_normalization=test_samplewise_std_normalization,
                    zca_whitening=test_zca_whitening, zca_epsilon=test_zca_epsilon, rotation_range=test_rotation_range, width_shift_range=test_width_shift_range,
                    height_shift_range=test_height_shift_range, brightness_range=test_brightness_range, shear_range=test_shear_range, zoom_range=test_zoom_range,
                    channel_shift_range=test_channel_shift_range, fill_mode=test_fill_mode, cval=test_cval,
                    horizontal_flip=test_horizontal_flip, vertical_flip=test_vertical_flip, rescale=test_rescale,
                    preprocessing_function=test_preprocessing_function, data_format=test_data_format, validation_split=test_validation_split, dtype=test_dtype)

      training_set = train_datagen.flow_from_directory(train_path,
                                               target_size=target_size,
                                               batch_size=batch_size,
                                               class_mode=class_mode)

      test_set = test_datagen.flow_from_directory(valid_path,
                                          target_size=target_size,
                                          batch_size=batch_size,
                                          class_mode=class_mode)

      history = final_model.fit_generator(training_set,
                      validation_data=test_set,
                      epochs=epochs,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set),
                      callbacks=callable_list,
                      validation_freq=validation_freq,
                      class_weight=class_weight, 
                      max_queue_size=max_queue_size, 
                      workers=workers, 
                      use_multiprocessing=use_multiprocessing,
                      shuffle=shuffle, initial_epoch=initial_epoch)

      return history, final_model
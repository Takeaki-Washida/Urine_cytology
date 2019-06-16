import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np

img_width, img_height = 150, 150
train_data_dir = 'data/train2'
validation_data_dir = 'data/validation2'

#steps_per_epoch：訓練パラメータ。 訓練データ数/batch_size の値がポピュラー。
steps_per_epoch = 20
#validation_steps：訓練パラメータ。 テストデータ数/batch_size の値がポピュラー。
validation_steps = 20
#epoch：訓練パラメータ。 1つの訓練データを何回繰り返し学習するかを表す。
epoch = 20

result_dir = 'results'
classes = ['good','daut','bad']

nb_classes = len(classes)

if __name__ == '__main__':
   
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    model.summary()

    # 最後のconv層の直前までの層をfreeze
    # 最終畳み込み層及び全結合層を再学習する。これがfine-tuning
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        classes=classes,
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        classes=classes,
        batch_size=32,
        class_mode='categorical')

    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        nb_epoch=20,
        validation_data=validation_generator,
        validation_steps=20)

    model.save_weights(os.path.join(result_dir, 'finetuning_0509.h5'))

import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.4, # 最大値の40%まで
        allow_growth=True # True->必要になったら確保, False->全部
    )
)
sess = sess = tf.Session(config=config)

"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)
"""


result_dir = 'results'
conv_name = 'block5_conv3'
classes = ['good','daut','bad']
nb_classes = len(classes)


# Define model here ---------------------------------------------------
def build_model():
    
    # VGG16
    input_tensor = Input(shape=(150, 150, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC
    fc = Sequential()
    fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
    fc.add(Dense(256, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=fc(vgg16.output))
    
    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'finetuning_0509.h5'))
    
    return model

H, W = 150, 150 # Input shape, defined by the model (model.input_shape)

# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (H, W), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams


def compute_saliency(model, guided_model, img_path, layer_name=conv_name, cls=-1, visualize=False, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    #--------- slide image get --------------
    ori_img = Image.open(img_path)

    #define slide range
    slide_xl = 0
    slide_xr = 100
    slide_yu = 0
    slide_yd = 100
    name_cnt_int = 1

    for m in range(9):
        for i in range(9):
            slide_img = ori_img.crop((slide_xl,slide_yu,slide_xr,slide_yd))
            name_cnt_str = str(name_cnt_int)
            roop_str = str(m)
            slide_name = './slide_img/slide_img_' + roop_str + '_' + name_cnt_str +  '.jpg'
            slide_img.save(slide_name)
            preprocessed_input = load_image(slide_name)

            pred = model.predict(preprocessed_input)[0]
                #print(pred)
            top_n = 3
            top_indices = pred.argsort()[-top_n:][::-1]
            result = [(classes[i], pred[i]) for i in top_indices]
            #print("number: ",name_cnt_str)
            print("number:",roop_str,name_cnt_str)
            print("xrange: ",slide_xl,slide_xr)
            print("yrange: ",slide_yu,slide_yd)
            for x in result:
                print(x)

            if cls == -1:
                cls = np.argmax(pred)
            
            print("argmax:",cls)
            if cls == 1:
                print("\n")
                print("-----Careful-----")
                print("-----Doubt spotted-----")
                print("\n")

            if cls == 2:
                print("\n")
                print("-----Warning!!!-----")
                print("-----Bad spotted!!!!!-----")
                print("\n")

            gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
            gb = guided_backprop(guided_model, preprocessed_input, layer_name)
            guided_gradcam = gb * gradcam[..., np.newaxis]
            cls = -1

            if save:
                cam_name = './cam_image/' + roop_str + '_' + name_cnt_str + '.jpg'
                jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
                jetcam = (np.float32(jetcam) + load_image(slide_name, preprocess=False)) / 2
                cv2.imwrite(cam_name, np.uint8(jetcam))
                    #cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
                    #cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
                
            name_cnt_int = int(name_cnt_str)
            name_cnt_int += 1
            #x軸スライド幅
            slide_xl += 50
            slide_xr += 50
            
            
            if visualize:
                    
                plt.figure(figsize=(15, 10))
                plt.subplot(131)
                plt.title('GradCAM')
                plt.axis('off')
                plt.imshow(load_image(img_path, preprocess=False))
                plt.imshow(gradcam, cmap='jet', alpha=0.5)

                plt.subplot(132)
                plt.title('Guided Backprop')
                plt.axis('off')
                plt.imshow(np.flip(deprocess_image(gb[0]), -1))
                    
                plt.subplot(133)
                plt.title('Guided GradCAM')
                plt.axis('off')
                plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
                plt.show()

        #右端までスライド完了、ｙ軸方向へスライド
        name_cnt_int = 0
        slide_xl = 0
        slide_xr = 100
        slide_yu = slide_yu + 50
        slide_yd = slide_yd + 50
    
  

    return gradcam, gb, guided_gradcam
    

if __name__ == '__main__':
    model = build_model()
    guided_model = build_guided_model()
    gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, layer_name=conv_name,
                                             img_path=sys.argv[1], cls=-1, visualize=False, save=True)

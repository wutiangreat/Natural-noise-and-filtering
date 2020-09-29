#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from keras.applications import vgg16
from keras.preprocessing import image
from keras.activations import relu, softmax
import keras.backend as K
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# Load VGG-16 model pretrained on ImageNet dataset
model = vgg16.VGG16(weights='imagenet')
# Get current session (assuming tf backend)
sess = K.get_session()

imgs=sys.argv[1]
img_path = '.\\image\\' + imgs
img = image.load_img(img_path, target_size=(224,224))
plt.imshow(img)
plt.grid('off')
plt.axis('off')

# Create a batch and preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = vgg16.preprocess_input(x)

# Get the initial predictions
preds = model.predict(x)
initial_class = np.argmax(preds)
print('Predicted:', vgg16.decode_predictions(preds, top=3)[0])

# Inverse of the preprocessing and plot the image
def plot_img(x,img_name,save_path):
    """
    x is a BGR image with shape (? ,224, 224, 3) 
    """
    tmp=1
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    plt.imshow(np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255)
    plt.grid('off')
    plt.axis('off')
    plt.gcf().set_size_inches(224 / 300, 224 / 300)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.savefig(save_path + '\\' + img_name + '.jpg',dpi=300,pad_inches=0,bbox_inches='tight')
    tmp=tmp+1

# Initialize adversarial example with input image
x_adv = x
# Added noise
x_noise = np.zeros_like(x)

# Set variables
epochs = 400
epsilon = 0.01
prev_probs = []
# One hot encode the initial class
target = K.one_hot(initial_class, 1000)

for i in range(epochs): 
    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})
    preds = model.predict(x_adv)

    # Store the probability of the target class
    prev_probs.append(preds[0][initial_class])

    if i%20==0:
        print(i, preds[0][initial_class], vgg16.decode_predictions(preds, top=3)[0])
plot_img(x_adv,imgs[:-4] + '_adv','.\image_adv')
plot_img(x_adv-x,imgs[:-4] + '_noise','.\\noise')
#plt.plot(np.arange(0,len(prev_probs)), prev_probs)
#plt.show()
#plt.savefig("SVD"+str(i)+".png", dpi=300,pad_inches=0,bbox_inches='tight')
np.save('.' + '\\' + 'npy' + '\\' + imgs[:-4], x)
np.save('.' + '\\' + 'npy_adv' + '\\' + imgs[:-4] + '_adv', x_adv)


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# load the model
new_model = load_model('cat_dog_100epochs.h5')

# load the img and convert it to a np array
dog_file = 'dog_image.jpg'
dog_img = image.load_img(dog_file, target_size=(150, 150))
dog_img = image.img_to_array(dog_img)
dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img/255

# print the predicted class along with the confidence
print(new_model.predict_classes(dog_img))
print(new_model.predict(dog_img))

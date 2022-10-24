import os
import tensorflow as tf


path = os.path.abspath('D:\\learn\\IKIT\\PI\\7-semestr\\teh-anal-dann\\2\\pr2')
test_data = os.path.join(path, 'data', 'prepared_data', 'test')
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data_generator=testdata_generator.flow_from_directory(
    test_data,
    batch_size=10,
    class_mode="categorical",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=False
)

test_number=test_data_generator.samples

model = tf.keras.models.load_model('model/my_model.h5')
res = model.evaluate(
    test_data_generator,
    batch_size=10
)
file = open("test.txt", "w")
file.write(res[0] + ' ' + res[1])

import os
import tensorflow as tf

path = os.path.abspath('D:\\learn\\IKIT\\PI\\7-semestr\\teh-anal-dann\\2\\prac2')
train_data = os.path.join(path, 'data', 'prepared_data', 'train')
valid_data = os.path.join(path, 'data', 'prepared_data', 'val')
test_data = os.path.join(path, 'data', 'prepared_data', 'test')

batch_size = 10
epochs = 1


traindata_generator=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, 
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)
validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_generator=traindata_generator.flow_from_directory(
    train_data,
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=True 
)
test_data_generator=testdata_generator.flow_from_directory(
    test_data,
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=False
)
valid_data_generator=validdata_generator.flow_from_directory(
    valid_data,
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=True
)

train_number=train_data_generator.samples
valid_number=valid_data_generator.samples

from keras.layers import Input, Add, Dense, Concatenate, AvgPool2D, Dropout,BatchNormalization,  GlobalAveragePooling2D
from keras import regularizers
from tensorflow.keras.models import Model

dense121_model= tf.keras.applications.densenet.DenseNet121(weights='imagenet',include_top=False, input_shape=(224,224, 3))
x= dense121_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x) 
x= Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x) 

x= Dropout(0.5)(x)
prediction= Dense(2, activation = 'softmax')(x)
model= Model(inputs= dense121_model.input, outputs= prediction)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks  import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

tensor_board=TensorBoard(log_dir="logs")
check_point=ModelCheckpoint("denseNet121.h5",monitor="val_accuracy",mode="auto",verbose=1,save_best_only=True)
reduce_lr=ReduceLROnPlateau(monitor="val_accuracy",factor=0.3,patience=50,min_delta=0.001,mode="auto",verbose=1)
history= model.fit(
    train_data_generator, 
    steps_per_epoch=train_number//batch_size, 
    validation_data= valid_data_generator, 
    validation_steps= valid_number//batch_size,
    shuffle=True, 
    epochs = epochs, 
    batch_size = batch_size,
    callbacks=[tensor_board,check_point,reduce_lr]
)
import shutil

try :
    shutil.rmtree('model')
except :
    print('папка model уже удалена')

os.mkdir('model')
model.save('model/my_model.h5')

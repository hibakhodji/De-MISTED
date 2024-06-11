import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import UpSampling2D, Dense, MaxPool2D, Flatten, BatchNormalization, Conv2D, Input, Dropout, Activation, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

outputDir=sys.argv[1]

# Data processing

train_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)
test_datagen  = ImageDataGenerator(rescale=1/255)


datapath="DATASET/"

train_generator = train_datagen.flow_from_directory(
        datapath+'train/',
        batch_size=32,
        target_size=(224,1024),
        shuffle = True,
        class_mode = 'binary')

validation_generator = valid_datagen.flow_from_directory(
        datapath+'valid/',
        batch_size=32,
        target_size=(224,1024),
        class_mode = 'binary',
        shuffle= False)


test_generator = test_datagen.flow_from_directory(
        datapath+'test/',
        batch_size=32,
        target_size=(224,1024),
        class_mode = 'binary',
        shuffle=False
)

# Check labels
print(test_generator.class_indices)

# Convolutional block  
def conv_block(inp, size):
    x = inp
    x = Conv2D(size, 5, padding='same')(x) 
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2,2)(x)
    x = Dropout(0.4)(x)
    return x


# Model creation
def createModel(placeholder, output_size):
    x = placeholder
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    x = Dense(output_size, activation='sigmoid')(x)
    return x



# Model input 
placeholder = Input([224, 1024, 3])
square = Model(inputs=placeholder,outputs=createModel(placeholder, 1))

square.summary()

# Config


numberOfEpochs  = 1 #100
optimizer       = tf.keras.optimizers.Adam(lr=1e-2, decay = 1e-4)
loss            = 'binary_crossentropy'
metrics         = ['accuracy',                tf.keras.metrics.Precision(),               tf.keras.metrics.Recall(),                  tf.keras.metrics.FalsePositives(),                tf.keras.metrics.FalseNegatives(),                tf.keras.metrics.TruePositives(),                 tf.keras.metrics.TrueNegatives()]
es_callback = EarlyStopping(monitor='val_loss', patience=10)
mcp_callback = ModelCheckpoint('square_modelcheckpoint.h5', save_best_only=True)
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')



square.compile(optimizer, loss, metrics)

def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss'] 
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(outputDir+'/Accuracy.pdf', dpi=200)
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(outputDir+'/Loss.pdf', dpi=200)
    plt.show()

x,y = next(train_generator)
#### model training and evaluation
print("Training model...")
modelName = "square_model"
weights = outputDir+"/square_weights.h5"
cb  = [es_callback, mcp_callback, lr_callback]
noe = numberOfEpochs

#history=square.fit(
#        train_generator,
#        epochs=noe,
#        validation_data=validation_generator, callbacks=cb)

history=square.fit(
        x,y,
        epochs=noe,
        validation_data=(x,y),
        callbacks=cb)

plot(history)
print("Saving model parameters")
square.save_weights(weights)
square.save(outputDir+"/square_model.h5")


print("Testing model...")
#metrics = square.evaluate(test_generator)
metrics = square.evaluate(x,y)
f = open(outputDir+'/metrics_'+str(modelName)+'.txt', 'w')
f.write(str(metrics)+'\n')
f.close()
print("Final testing accuracy :" +str(metrics[1])+"\n")





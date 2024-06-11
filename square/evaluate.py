import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


outputDir=sys.argv[1]

datapath="DATASET/"
test_datagen  = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
        datapath+'test/',
        batch_size=32,
        target_size=(224,1024),
        class_mode = 'binary',
        shuffle=False
)

msa = load_model('square_model.h5')

metrics = ['accuracy', 
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.FalsePositives(),
                   tf.keras.metrics.FalseNegatives(),
                   tf.keras.metrics.TruePositives(),
                   tf.keras.metrics.TrueNegatives()]



labels = test_generator.classes
fnames = test_generator.filenames

metrics = msa.evaluate(test_generator)

modelName = "square_model"
f = open(outputDir+'/eval_metrics_'+str(modelName)+'.txt', 'w')
f.write(str(metrics)+'\n')
f.close()
print("Final testing accuracy :" +str(metrics[1])+"\n")

#test_generator.reset

y_pred = (msa.predict(test_generator) > 0.5).astype("int32")

f = open("true_errors.txt", "a")
fnames = test_generator.filenames
for fname, prediction, label in zip(fnames, y_pred, labels):
  if prediction == label:
     #{'errors': 0, 'no_errors': 1}
     if label == 0 and prediction == 0:
        f.write(str(fname)) 
        f.write('\n')
f.close()


print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)
print('Classification Report')
class_labels = list(test_generator.class_indices.keys())
cr = classification_report(test_generator.classes, y_pred, target_names=class_labels)
print(cr)
cr_filename= "classification_report.txt"
with open(cr_filename, 'w') as file:
    file.write(cr)

print("Classification report written to", cr_filename)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig(outputDir+'/confusion_matrix.pdf', dpi=200)
plt.show()

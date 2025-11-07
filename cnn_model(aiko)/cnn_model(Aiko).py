from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2
base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Fine-tune the base model

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Add L2 regularization
    Dropout(0.5),  # Add dropout
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training and validation data
train = datagen.flow_from_directory('N:\\College\\Project\\Only Eyes', target_size=(64,64), color_mode='rgb',
                                    batch_size=32, class_mode='binary', subset='training')
val = datagen.flow_from_directory('N:\\College\\Project\\Only Eyes', target_size=(64,64), color_mode='rgb',
                                  batch_size=32, class_mode='binary', subset='validation')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model
history = model.fit(train, validation_data=val, epochs=25, callbacks=[early_stopping, lr_scheduler])

# Save the model in HDF5 format
model.save('cnn_model.h5')

# Save the model in SavedModel format
model.save(r'N:\\College\\Project\\Required\\saved_model.keras')  # Save the model in a directory named 'saved_model'

# Visualize the CNN architecture
plot_model(model, to_file='cnn_model_graph.png', show_shapes=True, show_layer_names=True)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
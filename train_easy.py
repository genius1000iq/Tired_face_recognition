import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# Упрощённая архитектура CNN (~40K параметров)
def create_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(227, 227, 1),
               kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(32, (3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', AUC(name='auc')])
    return model

# Аугментация изображений
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Генераторы
train_generator = train_datagen.flow_from_directory(
    'dataset_split/train',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    'dataset_split/val',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

# Проверка формы данных
x_batch, y_batch = next(train_generator)
print("Batch shape:", x_batch.shape, "| Labels shape:", y_batch.shape)

# Модель
model = create_model()

# Callbacks
early_stop = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max')
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)

# Тренировка
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

def get_pretrained_model_tf():
    # Load the pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers in the base model
    base_model.trainable = False

    # Add custom classification layers for 4 classes
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(4, activation='softmax')(x)

    # Build the final model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model

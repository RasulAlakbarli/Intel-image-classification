from keras.preprocessing.image import ImageDataGenerator


def  image_data_generator_train(train_dir):
    # Setting parameters for image augmentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # Generating datasets and labels from directories
    train_generator = train_datagen.flow_from_directory(
        directory = train_dir,
        target_size = (150, 150),
        color_mode = 'rgb', 
        class_mode='categorical', 
        batch_size=32,
        shuffle=True,
        seed=42)
    
    return train_generator


def  image_data_generator_valid(valid_dir):
    # Setting parameters for image augmentation
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Generating datasets and labels from directories
    valid_generator = valid_datagen.flow_from_directory(
        directory = valid_dir,
        target_size=(150, 150),
        color_mode='rgb',
        class_mode='categorical', 
        batch_size=32,
        shuffle=True,
        seed=42)
    
    return valid_generator


def  image_data_generator_test(test_dir):
    # Setting parameters for image augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generating datasets and labels from directories
    test_generator = test_datagen.flow_from_directory(
        directory = test_dir,
        target_size=(150, 150),
        color_mode='rgb',
        class_mode=None, 
        batch_size=1,
        shuffle=False,
        seed=42)
    
    return test_generator


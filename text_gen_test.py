from text_gen_functions import *


def main():
    """
    Driver function
    """
    # Define a variable "ul" which points to a file that contains the text
    ul = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

    # Use the "get_file" function from keras.utils module to download the text
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', ul)

    # Read the contents of the downloaded file by opening it in binary mode
    shakespeare_text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # File is decoded with 'utf-8' which encodes characters as 8-bit integers
    # This ensures compatibility and correct handling of non-ASCII characters

    # Preprocess the text returns char2idx, idx2char, vocab
    char_dict, char_arr, vocab = preprocess(shakespeare_text)

    # Convert the text to integers using the preprocessed character dictionary
    text_as_int = text_to_int(shakespeare_text, char_dict)

    # Generate sequences of text
    sequences = make_sequences(shakespeare_text, 100, text_as_int)

    # Divide the sequences into input and target sequences
    dataset = sequences.map(split_input_target)

    # Hyperparameters
    BATCH_SIZE = 64
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    BUFFER_SIZE = 10000

    # Shuffle the dataset and then divide them into batches
    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Build the model with the specified hyperparameters
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

    # Compile the model using the Adam optimizer and the sparse categorical crossentropy loss function
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Create a callback for saving checkpoints during the training process
    checkpoint_callback = create_checkpoint_callback(checkpoint_dir)

    # Train the model and save the checkpoints
    history = model.fit(data, epochs=80, callbacks=[checkpoint_callback])

    # Save the entire model to a HDF5 file
    model.save('model.h5')

    # Load the entire model from a HDF5 file
    model = tf.keras.models.load_model('model.h5')

    # Build a new model with batch size 1
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    # Load the weights of the most recent checkpoint
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    # Build the model with the specified input shape
    model.build(tf.TensorShape([1, None]))

    # Set the input text
    inp = "GLOUCESTER"

    # Generate text based on the input text
    generated_text = generate_text(model, inp, char_dict, char_arr)

    # Print the generated text
    print(generated_text)


if __name__ == "__main__":
    main()

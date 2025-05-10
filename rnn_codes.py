import tensorflow as tf
import numpy as np
import string

if __name__ == "__main__":
  inputs = np.array([
      ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
      ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
      ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
      ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
      ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
  ])

  expected = np.array([
      ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
      ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
      ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"], 
      ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
      ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
  ])
  
  # Encode strings to int indexes
  input_encoded = np.vectorize(string.ascii_uppercase.index)(inputs)
  input_encoded = input_encoded.astype(np.float32)
  one_hot_inputs = tf.keras.utils.to_categorical(input_encoded)

  expected_encoded = np.vectorize(string.ascii_uppercase.index)(expected)
  expected_encoded = expected_encoded.astype(np.float32)
  one_hot_expected = tf.keras.utils.to_categorical(expected_encoded)

  rnn = tf.keras.layers.SimpleRNN(128, return_sequences=True)

  model = tf.keras.Sequential(
      [
          rnn,
          tf.keras.layers.Dense(len(string.ascii_uppercase)),
      ]
  )

  model.compile(loss="categorical_crossentropy", optimizer="adam")

  model.fit(one_hot_inputs, one_hot_expected, epochs=10)

  new_inputs = np.array([["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"]])
  new_inputs_encoded = np.vectorize(string.ascii_uppercase.index)(new_inputs)
  new_inputs_encoded = new_inputs_encoded.astype(np.float32)
  new_inputs_encoded = tf.keras.utils.to_categorical(new_inputs_encoded)
  
  # Make prediction
  prediction = model.predict(new_inputs_encoded)

  # Get prediction of last time step
  prediction = np.argmax(prediction[0][-1])
  print(prediction)
  print(string.ascii_uppercase[prediction])
  
  #implwmnting same using LSTMs
  import tensorflow as tf
import numpy as np
import string

if __name__ == "__main__":
    inputs = np.array([
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        list("ZYXWVUTSRQPONMLKJIHGFEDCBA"),
        list("BDFHJLNPRTVXZACEGIKMOQSUWY"),
        list("MNOPQRSTUVWXYZABCDEFGHIJKL"),
        list("HGFEDCBALKJIPONMUTSRQXWVZY")
    ])

    expected = np.array([
        list("BCDEFGHIJKLMNOPQRSTUVWXYZA"),
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        list("CEGIKMOQSUWYABDFHJLNPRTVXZ"),
        list("NOPQRSTUVWXYZABCDEFGHIJKLM"),
        list("IJLKMNOPQRSTUVWXYZABCDEFGH")
    ])

    # Encode characters to integers (A=0, ..., Z=25)
    input_encoded = np.vectorize(string.ascii_uppercase.index)(inputs).astype(np.float32)
    expected_encoded = np.vectorize(string.ascii_uppercase.index)(expected).astype(np.float32)

    # One-hot encode the inputs and expected outputs
    one_hot_inputs = tf.keras.utils.to_categorical(input_encoded, num_classes=26)
    one_hot_expected = tf.keras.utils.to_categorical(expected_encoded, num_classes=26)

    # Build LSTM-based model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(26, 26)),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(one_hot_inputs, one_hot_expected, epochs=10)

    # Predict on a new input sequence
    new_inputs = np.array([list("BCDEFGHIJKLMNOPQRSTUVWXYZA")])
    new_encoded = np.vectorize(string.ascii_uppercase.index)(new_inputs).astype(np.float32)
    new_one_hot = tf.keras.utils.to_categorical(new_encoded, num_classes=26)

    prediction = model.predict(new_one_hot)

    # Decode prediction of last time step
    last_prediction_index = np.argmax(prediction[0][-1])
    print("Predicted next character:", string.ascii_uppercase[last_prediction_index])

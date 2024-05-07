import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import tensorflow as tf
import keras as k
import random

data_file = open('intents_spanish.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words=[]
classes =[]
documents=[]
ignore_words=['?', '!']

# Recorre cada interacion y sus patrones en el archivo JSON
for intent in intents['intents']:
    for patterns in intent['patterns']:
        # Tokeniza las palabras en cada patron y las agrega a la lista de palabras
        w = nltk.word_tokenize(patterns)
        words.extend(w)

        # Agregq el par (patron,etiqueta) a la lista de documentos
        documents.append((w, intent['tag']))

        # Si la etiqueta no está en la lista de clase, la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematiza las alabras y las convierte en minuisculas, excluyendo las palabras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Guarda las listas de palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Crea el conjunto de entrenamiento
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        # Crea una bolsa de palabras binarias para cada patrón
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    # Crea un vector salida con un 1 en la posicion correspondiente a la etiqueta de la intencion
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezcla aleatoriamente el conjunto de entrenamiento
random.shuffle(training)

# Divide el conjunto de entrenamiento en caracteristicas (train_x) y etiquetas (train_y)
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# Crea el modelo de red neuronal
model = k.Sequential()
model.add(k.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(k.layers.Dropout(0,5))
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dropout(0,5))
model.add(k.layers.Dense(len(train_y[0]), activation='softmax'))

# Configurar el optimizador con una tasa de aprendizaje exponencialmente decreciente
lr_schedule = k.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)

sgd = k.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrena el modelo con el conjunto de entrenamiento
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guarda el modelo entrenado en un archgivo h5
model.save('ChatBot_model.h5', hist)

print("Modelo Creado")
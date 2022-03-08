import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import preprocessing

class IntentModel:
    def __init__(self, model_name, proprocess):
        self.labels = {0: "casual", 1: "park"}
        self.model = load_model()
        self.model = load_model(model_name)

        self.p = proprocess

    def predict_class(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        from OOD.config.GlobalParams import Max_SEQ_LEN
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=Max_SEQ_LEN, padding='post')
        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
import logging
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
class cbf:
    def on_train_begin(self, x):
        self.cnte = 0
        self.preloss = 0
        self.los = []
        print('begin training...', time.time()-tm)
    def on_epoch_begin(self, x):
        self.cnte += 1
        print('epoch begin', self.cnte, x.running_training_loss, time.time()-tm)
    def on_epoch_end(self, x):
        self.los.append(x.running_training_loss-self.preloss)
        print('epoch end', self.cnte, 'loss:', self.los[-1], time.time()-tm)
        self.preloss = x.running_training_loss
    def on_train_end(self, x):
        print('end train', time.time()-tm)
        print('total loss:', x.running_training_loss)
        print('loss:', self.los)
        
tm = time.time()
print('start')
mdl = Word2Vec(corpus_file='test.txt', size=300, workers=10, compute_loss=True, iter=50, callbacks=(cbf(),)) #, iter=5, min_count=10

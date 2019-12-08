import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
class cbf:
    def on_train_begin(self, x):
        self.cnte = 0
        print('begin training...', time.time()-tm)
    def on_epoch_begin(self, x):
        self.cnte += 1
		self.cntb = 0
        print('epoch begin', self.cnte, time.time()-tm)
    def on_batch_begin(self, x):
		return
    def on_batch_end(self, x):
		self.cntb += 1
		if(self.cntb%100==0):
			print('bat:',self.cnte,self.cntb,time.time()-tm)
    def on_epoch_end(self, x):
        print('\nepoch end', self.cnte, time.time()-tm)
    def on_train_end(self, x):
        print('end train', time.time()-tm)
        
tm = time.time()
mdl = Word2Vec(corpus_file='input.txt', size=300, workers=10, callbacks=(cbf(),)) #, iter=5, min_count=10, compute_loss=False
mdl.save('zhvec')


class getdata(object):
    
    def __init__(self, images, labels, one_hot = True):
        assert images.shape[0] == labels.shape[0]
        self.num_examples = images.shape[0]
        assert images.shape[3] == 1
        self.images = images
        self.one_hot = one_hot
        self.labels = labels
        if not one_hot:
            self.labels = onehot(labels)
        self.epoch = 0
        self.startofepoch = 0
        
    def nextbatch(self, batch_size):
        start = self.startofepoch
        end = start + batch_size
        
        if end > self.num_examples:
            self.epoch += 1
            
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            
            start = 0
            end = batch_size
            
        self.startofepoch = end
        return self.images[start:end], self.labels[start:end]
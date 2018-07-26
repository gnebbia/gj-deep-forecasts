import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

loss = []
acc = []
ifip_acc = {}
for e in tf.train.summary_iterator("/Users/derek/Desktop/events.out.tfevents.1532561566.lambda-quad"):
    for v in e.summary.value:
        if v.tag == 'Train/loss':
        	loss.append(v.simple_value)
        elif v.tag == 'Train/acc':
        	acc.append(v.simple_value)
        elif v.tag.startswith('IFIP') and v.tag.endswith('acc'):
        	if v.tag[:-4] in ifip_acc.keys():
        		ifip_acc[v.tag[:-4]].append(v.simple_value)
        	else:
        		ifip_acc[v.tag[:-4]] = [v.simple_value]

plt.figure(1)
plt.subplot(211)
plt.title("Training Loss")
plt.plot(loss, color="red")
plt.xlim([0,500])
plt.subplot(212)
plt.title("Training Accuracy")
plt.plot(acc, color = "blue")
plt.xlim([0,500])
plt.xlabel('Minibatch Iterations')
plt.subplots_adjust(hspace=0.3)
plt.show()

plt.figure(2)
epoch_1 = []
epoch_2 = []
epoch_3 = []
for x in ifip_acc.keys():
	epoch_1.append(ifip_acc[x][0])
	epoch_2.append(ifip_acc[x][1])
	if len(ifip_acc[x]) > 2:
		epoch_3.append(ifip_acc[x][2]) 
n, bins, patches = plt.hist(epoch_1, 15, density=1, facecolor="green", edgecolor="black", alpha=0.3, label="Epoch 1")
n, bins, patches = plt.hist(epoch_2, 15, density=1, facecolor="blue", edgecolor="black", alpha=0.3, label="Epoch 2")
n, bins, patches = plt.hist(epoch_3, 15, density=1, facecolor="red", edgecolor="black", alpha=0.3, label="Epoch 3ev")
plt.legend()
plt.show()